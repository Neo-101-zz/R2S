import datetime
import logging
import time

from global_land_mask import globe
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import mapper
from netCDF4 import Dataset
import numpy as np
import pygrib
from scipy import interpolate
import pandas as pd

import utils
import satel_scs
import era5

Base = declarative_base()

class matchManager(object):

    def __init__(self, CONFIG, period, region, basin, passwd, save_disk):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.save_disk = save_disk
        self.engine = None
        self.session = None
        self.basin = basin

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        # self.load_match()
        # breakpoint()

        self.extract()

    def load_match(self):
        Match = utils.create_match_table(self, 'sfmr', 'era5')
        match_list = []
        for i in range(len(self.match_data_sources)):
            df_row = self.match_data_sources.iloc[i]
            row = Match()
            row.tc_sid = df_row['TC_sid']
            row.date_time = df_row['datetime'].to_pydatetime()
            row.match = df_row['match']
            row.tc_sid_datetime = f'{row.tc_sid}_{row.date_time}'

            match_list.append(row)

        utils.bulk_insert_avoid_duplicate_unique(
            match_list, self.CONFIG['database']['batch_size']['insert'],
            Match, ['tc_sid_datetime'], self.session,
            check_self=True)

    def extract(self):
        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name'][
            self.basin]
        IBTrACS = utils.get_class_by_tablename(self.engine, table_name)
        tc_query = self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        )
        total = tc_query.count()
        # Traverse WP TCs
        for idx, tc in enumerate(tc_query):
            try:
                converted_lon = utils.longitude_converter(
                    tc.lon, '360', '-180')
                if bool(globe.is_land(tc.lat, converted_lon)):
                    continue
                if tc.date_time.minute or tc.date_time.second:
                    continue
                if idx < total - 1:
                    next_tc = tc_query[idx + 1]
                    # This TC and next TC is same TC
                    if tc.sid == next_tc.sid:
                        self.extract_between_two_tc_records(tc, next_tc)
                    # This TC differents next TC
                    else:
                        pass
                else:
                    pass
            except Exception as msg:
                breakpoint()
                exit(msg)

    def extract_between_two_tc_records(self, tc, next_tc):
        """
        Notes
        -----
        In this project, the range of longitude is from 0 to 360, so
        there may be data leap near the prime merdian.  And the
        interpolation should be ajusted.

        """
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are far away in time
        if delta.days:
            return
        hours = int(delta.seconds / 3600)
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are too close in time
        if not hours:
            return

        Match = utils.create_match_table(self, 'sfmr', 'era5')
        hit_dt = []
        match_dt = []
        for h in range(hours):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)
            same_sid_dt_query = self.session.query(Match).filter(
                Match.date_time == interped_tc.date_time,
                Match.tc_sid == interped_tc.sid)
            same_sid_dt_count = same_sid_dt_query.count()

            if not same_sid_dt_count:
                continue
            elif same_sid_dt_count == 1:
                hit_dt.append(interped_tc.date_time)
                if same_sid_dt_query[0].match:
                    match_dt.append(interped_tc.date_time)
            else:
                self.logger.error((f"""Strange: two or more """
                                   f"""comparison has same sid """
                                   f"""and datetime"""))
                breakpoint()
                exit()

        hit_count = len(hit_dt)
        match_count = len(match_dt)
        if hit_count == hours and not match_count:
            print((f"""[Skip] All internal hours of TC """
                   f"""{tc.name} between {tc.date_time} """
                   f"""and {next_tc.date_time}"""))
            return

        # Check existence of SFMR between two IBTrACS records
        existence, spatial_temporal_info = utils.sfmr_exists(
            self, tc, next_tc)
        if not existence:
            # First executed here between particular two TCs
            if hit_count < hours:
                # update match of data sources
                utils.update_no_match_between_tcs(self, Match, hours,
                                                  tc, next_tc)

            print((f"""[Not exist] SFMR of TC {tc.name} between """
                   f"""{tc.date_time} and {next_tc.date_time}"""))
            return

        # Round SMFR record to different hours
        # hour_info_pt_idx:
        #   {
        #       hour_datetime_1: {
        #           info_idx_1: [pt_idx_1, pt_idx_2, ...],
        #           info_idx_2: [pt_idx_1, pt_idx_2, ...],
        #           ...
        #       }
        #       hour_datetime_2: {
        #           ...
        #       }
        #       ...
        #   }
        hour_info_pt_idx = utils.sfmr_rounded_hours(
            self, tc, next_tc, spatial_temporal_info)
        if not len(hour_info_pt_idx):
            # First executed here between particular two TCs
            if hit_count < hours:
                # update match of data sources
                utils.update_no_match_between_tcs(self, Match, hours,
                                                  tc, next_tc)
            print((f"""[Fail rounding to hour] SFMR of TC {tc.name} """
                   f"""between {tc.date_time} and """
                   f"""{next_tc.date_time}"""))
            return

        if hit_count == hours:
            self.extract_with_all_hours_hit(tc, next_tc, hours, match_dt,
                                            spatial_temporal_info,
                                            hour_info_pt_idx)
        else:
            self.extract_with_not_all_hours_hit(tc, next_tc, hours,
                                                spatial_temporal_info,
                                                hour_info_pt_idx)

    def extract_with_all_hours_hit(
        self, tc, next_tc, hours, match_dt, spatial_temporal_info,
        hour_info_pt_idx):
        #
        need_exit = False
        for h in range(hours):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)
            SFMRERA5 = self.create_sfmr_era5_table(interped_tc.date_time)

            if interped_tc.date_time not in match_dt:
                print((f"""[Skip] matching SFMR and ERA5 """
                       f"""around TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))
                continue

            sfmr_success, data, hourtimes, area = \
                    self.extract_sfmr_around_interped_tc(
                        spatial_temporal_info,
                        hour_info_pt_idx[interped_tc.date_time],
                        interped_tc)
            print((f"""[Redo] extract SFMR """
                   f"""around TC {interped_tc.name} """
                   f"""on {interped_tc.date_time}"""))
            if not sfmr_success:
                self.logger.error(
                    'Match True but fail extracting SFMR')
                continue

            try:
                data = utils.add_era5(self, 'sfmr', interped_tc, data,
                                      hourtimes, area)
            except Exception as msg:
                breakpoint()
                exit(msg)

            if data is None:
                # Exception occurs and do not need to save matchup of
                # data sources before shutdowning program.  Because all
                # hours between `tc` and `next_tc` are recored in `Match`
                # table
                need_exit = True
                break

            if not len(data):
                self.logger.error(
                    'Match True but fail adding ERA5')
                continue

            utils.bulk_insert_avoid_duplicate_unique(
                data, self.CONFIG['database']['batch_size']['insert'],
                SFMRERA5, ['sfmr_datetime_lon_lat'], self.session,
                check_self=True)

        if need_exit:
            exit(1)

    def extract_with_not_all_hours_hit(self, tc, next_tc, hours,
                                       spatial_temporal_info, 
                                       hour_info_pt_idx):
        need_exit = False
        Match = utils.create_match_table(self, 'sfmr', 'era5')

        for h in range(hours):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)
            SFMRERA5 = self.create_sfmr_era5_table(interped_tc.date_time)

            if interped_tc.date_time not in hour_info_pt_idx.keys():
                # update corrseponding match
                utils.update_one_row_of_match(self, Match, interped_tc,
                                              False)
                print((f"""[Not exist] SFMR of TC {tc.name} near """
                       f"""{interped_tc.date_time}"""))
                continue

            try:
                sfmr_success, data, hourtimes, area = \
                        self.extract_sfmr_around_interped_tc(
                            spatial_temporal_info,
                            hour_info_pt_idx[interped_tc.date_time],
                            interped_tc)
            except Exception as msg:
                breakpoint()
                exit(msg)

            if not sfmr_success:
                # Normal fail, need continue comparing
                utils.update_one_row_of_match(self, Match, interped_tc,
                                              False)
                print((f"""[Not found] SFMR """
                       f"""around TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))
                continue

            try:
                data = utils.add_era5(self, 'sfmr', interped_tc, data,
                                      hourtimes, area)
            except Exception as msg:
                breakpoint()
                exit(msg)

            if data is None:
                # Exception occurs
                need_exit = True
                break

            if not len(data):
                utils.update_one_row_of_match(self, Match, interped_tc,
                                              False)
                print((f"""[No matchup] SFMR and ERA5"""
                       f"""around TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))
                continue

            utils.update_one_row_of_match(self, Match, interped_tc, True)
            print((f"""[Match] SFMR and ERA5 """
                   f"""around TC {interped_tc.name} """
                   f"""on {interped_tc.date_time}"""))

            utils.bulk_insert_avoid_duplicate_unique(
                data, self.CONFIG['database']['batch_size']['insert'],
                SFMRERA5, ['sfmr_datetime_lon_lat'], self.session,
                check_self=True)

        if need_exit:
            exit(1)

    def extract_sfmr_around_interped_tc(
        self, sfmr_brief_info, one_hour_info_pt_idx, interped_tc):
        #
        data = []
        try:
            success, sfmr_tracks, sfmr_dts, sfmr_lons, sfmr_lats, \
                    sfmr_windspd = utils.average_sfmr_along_track(
                        self, interped_tc, sfmr_brief_info,
                        one_hour_info_pt_idx, use_slow_wind=True)

            if not success:
                return success, data, None, None

            hourtimes = set()
            area = None
            north = -90
            west = 360
            south = 90
            east = 0

            interped_tc_center = (
                interped_tc.lat,
                utils.longitude_converter(interped_tc.lon, '360', '-180'))
            SFMRERA5 = self.create_sfmr_era5_table(interped_tc.date_time)
            tracks_num = len(sfmr_tracks)
        except Exception as msg:
            breakpoint()
            exit(msg)

        try:
            for i in range(tracks_num):
                north = max(max(sfmr_lats[i]), north)
                west = min(min(sfmr_lons[i]), west)
                south = min(min(sfmr_lats[i]), south)
                east = max(max(sfmr_lons[i]), east)

                for j in range(len(sfmr_dts[i])):
                    row = SFMRERA5()
                    row.sid = interped_tc.sid
                    row.sfmr_datetime = sfmr_dts[i][j]
                    row.lon = sfmr_lons[i][j]
                    row.lat = sfmr_lats[i][j]

                    sfmr_pt = (row.lat, utils.longitude_converter(
                        row.lon,  '360', '-180'))

                    row.east_shift_from_center = \
                            utils.east_or_north_shift(
                                'east', interped_tc_center, sfmr_pt)
                    row.north_shift_from_center = \
                            utils.east_or_north_shift(
                                'north', interped_tc_center, sfmr_pt)

                    row.sfmr_datetime_lon_lat = (
                        f"""{row.sfmr_datetime}"""
                        f"""_{row.lon:.3f}_{row.lat:.3f}""")

                    row.sfmr_windspd = sfmr_windspd[i][j]

                    this_hourtime = utils.hour_rounder(
                        row.sfmr_datetime).hour
                    # Skip situation that hour is rounded to next day
                    if (row.sfmr_datetime.hour == 23
                        and this_hourtime == 0):
                        continue

                    # Strictest reading rule: None of columns is none
                    skip = False
                    for key in row.__dict__.keys():
                        if getattr(row, key) is None:
                            skip = True
                            break
                    if skip:
                        continue
                    else:
                        data.append(row)
                        hourtimes.add(this_hourtime)
        except Exception as msg:
            breakpoint()
            exit(msg)

        diff = 1.0
        area = [round(north + diff, 3),
                round((west - diff + 360) % 360, 3),
                round(south - diff, 3),
                round((east + diff + 360) % 360, 3)]

        return success, data, list(hourtimes), area

    def create_sfmr_era5_table(self, dt):
        table_name = utils.gen_tc_sfmr_era5_tablename(dt, self.basin)

        class SFMRERA5(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(SFMRERA5, t)

            return SFMRERA5

        cols = utils.get_basic_sfmr_era5_columns(tc_info=True)

        cols.append(Column('sfmr_windspd', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, False,
                                 self.save_disk, '', 'tc')
        era5_cols = era5_.get_era5_columns(tgt_name='sfmr')
        cols = cols + era5_cols

        cols.append(Column('era5_10m_neutral_equivalent_windspd',
                           Float, nullable=False))
        cols.append(Column('era5_10m_neutral_equivalent_winddir',
                           Float, nullable=False))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(SFMRERA5, t)

        self.session.commit()

        return SFMRERA5
