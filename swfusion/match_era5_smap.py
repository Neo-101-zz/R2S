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

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.lats = dict()
        self.lons = dict()
        self.spa_resolu = dict()

        self.spa_resolu['smap'] = self.CONFIG['rss'][
            'spatial_resolution']
        self.lats['smap'] = [
            y * self.spa_resolu['smap'] - 89.875 for y in range(720)]
        self.lons['smap'] = [
            x * self.spa_resolu['smap'] + 0.125 for x in range(1440)]

        self.spa_resolu['era5'] = self.CONFIG['era5'][
            'spatial_resolution']
        self.lats['era5'] = [
            y * self.spa_resolu['era5'] - 90 for y in range(721)]
        self.lons['era5'] = [
            x * self.spa_resolu['era5'] for x in range(1440)]

        self.edge = self.CONFIG['regression']['edge_in_degree']
        self.half_edge = self.edge / 2
        self.half_edge_grid_intervals = int(
            self.half_edge / self.spa_resolu['smap'])

        self.pres_lvls = self.CONFIG['era5']['pres_lvls']

        utils.setup_database(self, Base)

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.sources = ['era5', 'smap']

        self.extract()

    def extract(self):
        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name'][
            self.basin]
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)
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
                        success = self.extract_detail(tc)
                        self.info_after_extracting_detail(tc, success,
                                                          True)
                else:
                    success = self.extract_detail(tc)
                    self.info_after_extracting_detail(tc, success,
                                                      True)
            except Exception as msg:
                breakpoint()
                exit(msg)

    def info_after_extracting_detail(self, tc, success, update_match):
        Match = utils.create_match_table(self, 'smap', 'era5')

        if update_match:
            if success:
                utils.update_one_row_of_match(self, Match, tc, True)
                print((f"""[Match] SMAP and ERA5"""
                       f"""around TC {tc.name} """
                       f"""on {tc.date_time}"""))
            else:
                utils.update_one_row_of_match(self, Match, tc, False)
                print((f"""[Not match] SMAP and ERA5 """
                       f"""around TC {tc.name} near """
                       f"""{tc.date_time}"""))
        else:
            if success:
                print((f"""[Redo] match SMAP and ERA5"""
                       f"""around TC {tc.name} """
                       f"""on {tc.date_time}"""))
            else:
                self.logger.error((
                    f"""Match True but fail matching SMAP and ERA5"""
                    f"""around TC {tc.name} """
                    f"""on {tc.date_time}"""))

    def extract_between_two_tc_records(self, tc, next_tc):
        """
        Notes
        -----
        In this project, the range of longitude is from 0 to 360, so there
        may be data leap near the prime merdian.  And the interpolation
        should be ajusted.

        """
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are far away in time
        if delta.days:
            self.extract_detail(tc)
            return
        hours = int(delta.seconds / 3600)
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are too close in time
        if not hours:
            self.extract_detail(tc)
            return

        Match = utils.create_match_table(self, 'smap', 'era5')
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

        # Extract from the interval between two TC records
        if hit_count == hours:
            self.extract_with_all_hours_hit(tc, next_tc, hours, match_dt)
        else:
            self.extract_with_not_all_hours_hit(tc, next_tc, hours)

    def extract_with_all_hours_hit(self, tc, next_tc, hours, match_dt):
        for h in range(hours):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)

            if interped_tc.date_time not in match_dt:
                print((f"""[Skip] matching SMAP and ERA5 """
                       f"""around TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))
                continue

            success = self.extract_detail(interped_tc)
            self.info_after_extracting_detail(interped_tc, success,
                                              False)

    def extract_with_not_all_hours_hit(self, tc, next_tc, hours):
        for h in range(hours):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)

            success = self.extract_detail(interped_tc)
            self.info_after_extracting_detail(interped_tc, success,
                                              True)

    def extract_detail(self, tc):
        SMAPERA5 = utils.create_smap_era5_table(self, tc.date_time)
        # Skip this turn if there is no SMAP data around TC
        try:
            data, hourtimes, area = self.extract_smap(tc, SMAPERA5)
            if not len(data) or not len(hourtimes) or not len(area):
                return False
        except Exception as msg:
            breakpoint()
            exit(msg)

        try:
            data = utils.add_era5(self, 'smap', tc, data,
                                  hourtimes, area)
            if not len(data):
                return False
        except Exception as msg:
            breakpoint()
            exit(msg)

        utils.bulk_insert_avoid_duplicate_unique(
            data, self.CONFIG['database']['batch_size']['insert'],
            SMAPERA5, ['satel_datetime_lon_lat'], self.session,
            check_self=True)

        return True

    def get_square_around_tc(self, tc_lon, tc_lat):
        """Get indices of square corners around tropical cyclone center
        in grid.

        parameters
        ----------
        tc_lon: float
        tc_lat: float

        Return
        ------
        success: bool
        lat1_idx: int
        lat2_idx: int
        lon1_idx: int
        lon2_idx: int
        lat1: float
        lon1: float

        """
        try:
            tc_lon_in_grid, tc_lon_in_grid_idx = \
                utils.get_nearest_element_and_index(
                    self.lons['smap'], tc_lon)
            tc_lat_in_grid, tc_lat_in_grid_idx = \
                utils.get_nearest_element_and_index(
                    self.lats['smap'], tc_lat)
            lat1_idx = (tc_lat_in_grid_idx
                        - self.half_edge_grid_intervals)
            lat1 = self.lats['smap'][lat1_idx]
            lat2_idx = (tc_lat_in_grid_idx
                        + self.half_edge_grid_intervals)
            lat2 = self.lats['smap'][lat2_idx]

            if lat1 < -90 or lat2 > 90:
                success = False
            else:
                success = True

            lons_num = len(self.lons['smap'])
            lon1_idx = (tc_lon_in_grid_idx
                        - self.half_edge_grid_intervals)
            lon1_idx = (lon1_idx + lons_num) % lons_num
            lon1 = self.lons['smap'][lon1_idx]
            lon2_idx = (tc_lon_in_grid_idx
                        + self.half_edge_grid_intervals)
            lon2_idx = (lon2_idx + lons_num) % lons_num
            lon2 = self.lons['smap'][lon2_idx]

            # ATTENTION: if area crosses the prime merdian, e.g.
            # area = [10, 358, 8, 2], ERA5 API cannot retrieve
            # requested data correctly,  so skip this situation
            if lon1_idx >= lon2_idx:
                success = False

        except Exception as msg:
            breakpoint()
            exit(msg)

        return success, lat1_idx, lat2_idx, lon1_idx, lon2_idx,\
            lat1, lon1

    def extract_smap(self, tc, SMAPERA5):
        """Extract SMAP data according to tropical cyclone data
        from IBTrACS.

        parameters
        ----------
        tc_dt: datetime
            Datetime of tropical cyclone.  May be interpolated value,
            not originally from IBTrACS.
        tc_lon: float
            Longitude of tropical cyclone.  May be interpolated value,
            not originally from IBTrACS.
        tc_lat: float
            Latitude of tropical cyclone.  May be interpolated value,
            not originally from IBTrACS.
        SMAPERA5: table class
            Repersentation of SMAP data and matching ERA5 data around
            tropical cyclone.

        Return
        ------
        data: list
            List of extracted SMAP data.  Elements are SMAPERA5 type.
        hourtimes: list
            List of hour time that SMAP data is closest to.  e.g. 2
            o'clock and 5 o'clock.

        """
        try:
            satel_manager = satel_scs.SCSSatelManager(
                self.CONFIG, self.period, self.region,
                self.db_root_passwd,
                save_disk=self.save_disk, work=False)
            smap_file_path = satel_manager.download('smap',
                                                    tc.date_time)
            if smap_file_path is None:
                return [], None, None
            data, hourtimes, area = self.get_smap_part(
                SMAPERA5, tc, smap_file_path)
        except Exception as msg:
            breakpoint()
            exit(msg)

        # if tc.date_time == datetime.datetime(2015, 5, 7, 23, 0, 0):
        #     breakpoint()

        return data, hourtimes, area

    def get_smap_part(self, SMAPERA5, tc, smap_file_path):
        success, lat1_idx, lat2_idx, lon1_idx, lon2_idx, lat1, lon1 = \
                self.get_square_around_tc(tc.lon, tc.lat)
        if not success:
            return [], None, None

        dataset = Dataset(smap_file_path)
        # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which
        # faster than 1 m/s, so must disable auto mask
        dataset.set_auto_mask(False)
        vars = dataset.variables
        # Square around TC does not cross the prime meridian
        minute = vars['minute'][lat1_idx:lat2_idx+1,
                                lon1_idx:lon2_idx+1, :]
        wind = vars['wind'][lat1_idx:lat2_idx+1,
                            lon1_idx:lon2_idx+1, :]

        lats_num, lons_num, passes_num = minute.shape
        minute_missing = self.CONFIG['smap']['missing_value']['minute']
        wind_missing = self.CONFIG['smap']['missing_value']['wind']

        data = []
        lats = []
        lons = []
        north = -90
        west = 360
        south = 90
        east = 0
        hourtimes = set()

        for y in range(lats_num):
            lat_of_row = y * self.spa_resolu['smap'] + lat1

            for x in range(lons_num):
                lon_of_col = x * self.spa_resolu['smap'] + lon1
                lon_of_col = (lon_of_col + 360) % 360

                for i in range(passes_num):
                    try:
                        if (minute[y][x][i] == minute_missing
                            or wind[y][x][i] == wind_missing):
                            continue
                        if minute[y][x][0] == minute[y][x][1]:
                            continue
                        if minute[y][x][i] == 1440:
                            continue
                        time_ = datetime.time(
                            *divmod(int(minute[y][x][i]), 60), 0)
                        # Temporal window is one hour
                        pixel_dt = datetime.datetime.combine(
                            tc.date_time.date(), time_)
                        # MUST use abs() method
                        delta = abs(pixel_dt - tc.date_time)
                        if delta.days or delta.seconds > 1800:
                            continue

                        # SMAP originally has land mask, so it's not
                        # necessary to check whether each pixel is land
                        # or ocean
                        row = SMAPERA5()
                        row.sid = tc.sid
                        row.satel_datetime = pixel_dt
                        row.x = x - self.half_edge_grid_intervals
                        row.y = y - self.half_edge_grid_intervals

                        row.lon = lon_of_col
                        # if row.lon < west:
                        #     west = row.lon
                        # if row.lon > east:
                        #     east = row.lon
                        lons.append(row.lon)

                        row.lat = lat_of_row
                        # if row.lat < south:
                        #     south = row.lat
                        # if row.lat > north:
                        #     north = row.lat
                        lats.append(row.lat)

                        row.satel_datetime_lon_lat = (
                            f"""{row.satel_datetime}"""
                            f"""_{row.lon}_{row.lat}""")
                        row.smap_windspd = float(wind[y][x][i])

                        this_hourtime = utils.hour_rounder(
                            row.satel_datetime).hour
                        # Skip situation that hour is rounded to next
                        # day
                        if (row.satel_datetime.hour == 23
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

        if not len(data):
            return data, None, None

        north = max(lats)
        south = min(lats)
        east = max(lons)
        west = min(lons)
        # 'area' parameter to request ERA5 data via API:
        # North, West, South, East
        # e.g. [12.125, 188.875, 3.125, 197.875]
        #
        # Due to difference bewteen ERA5 and RSS grid, need
        # to expand the area a little to request an ERA5 grid
        # consists of all minimum squares around all RSS points.
        #
        # Considering the spatial resolution of ERA5 ocean waves
        # are 0.5 degree x 0.5 degree, not 0.25 x 0.25 degree of
        # atmosphere, we need to expand a little more, at least 0.5
        # degree
        diff = 0.5
        area = [north + diff, (west - diff + 360) % 360,
                south - diff, (east + diff + 360) % 360]
        # --------------------------------------------------
        # Updated 2019/12/24
        #
        # It seems that ERA5 will autonomically extract grid according
        # to user's input of 'area' parameter when requesting ERA5
        # data via API.
        #
        # For example, if 'area' parameter is
        # [1.125, 0.625, 0.125, 1.625], the ERA5 grib file's lats will
        # be [0.125, 0.375, 0.625, 0.875, 1.125] and its lons will be
        # [0.625, 0.875, 1.125. 1.375, 1.625], which is 0.25 degree
        # grid but the coordinates are not the multiple of 0.25.
        #
        # Becaues we consider ERA5 reanalysis grid coordinates are the
        # multiple of 0.25 before, so we will squeeze the expanded area
        # a little to a range which corners' coordiantes are the
        # multiple of 0.25.
        #
        # North, West, South, East
        for idx, val in enumerate(area):
            if utils.is_multiple_of(val, 0.125):
                # decrease north and east a little
                if not idx or idx == 3:
                    area[idx] = val - 0.125
                # increase west and south a little
                else:
                    area[idx] = val + 0.125
        area[1] = (area[1] + 360) % 360
        area[3] = (area[3] + 360) % 360

        return data, list(hourtimes), area

