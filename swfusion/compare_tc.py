import calendar
import datetime
import logging
import os
import pickle
import statistics
import string
import time

import matplotlib.pyplot as plt
from global_land_mask import globe
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import pygrib
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import or_
import netCDF4

import ccmp
import era5
import satel_scs
import utils

Base = declarative_base()

class TCComparer(object):

    def __init__(self, CONFIG, period, region, basin, passwd,
                 save_disk, compare_instructions):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.save_disk = save_disk
        self.engine = None
        self.session = None
        self.compare_instructions = list(set(compare_instructions))
        self.basin = basin

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.spa_resolu = dict()

        self.spa_resolu['ccmp'] = self.CONFIG['ccmp']\
                ['spatial_resolution']
        self.spa_resolu['era5'] = dict()
        self.spa_resolu['era5']['atm'] = self.CONFIG['era5']\
                ['spatial_resolution']
        self.spa_resolu['era5']['ocean'] = self.CONFIG['era5']\
                ['ocean_spatial_resolution']

        self.lats = dict()
        self.lons = dict()

        self.lats['era5'] = [
            y * self.spa_resolu['era5']['atm'] - 90 for y in range(721)]
        self.lons['era5'] = [
            x * self.spa_resolu['era5']['atm'] for x in range(1440)]

        self.reg_edge = self.CONFIG['regression']['edge_in_degree']
        self.half_reg_edge = self.reg_edge / 2
        self.half_reg_edge_grid_intervals = int(
            self.half_reg_edge / self.spa_resolu['era5']['atm'])

        self.pres_lvls = self.CONFIG['era5']['pres_lvls']

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.zorders = self.CONFIG['plot']['zorders']['compare']

        utils.reset_signal_handler()

        self.source_candidates = ['smap', 'era5', 'sfmr',
                                  'smap_prediction']
        self.sources = []
        for part in self.compare_instructions:
            if part in self.source_candidates:
                self.sources.append(part)

        # if 'smap' in self.sources and 'sfmr' in self.sources:
        #     self.logger.error((
        #         f"""Not support synchronously compare """
        #         f"""SMAP and SFMR with or without other """
        #         f"""sources"""))
        #     exit()

        # Swap 'sfmr' and first source string when necessary
        if 'sfmr' in self.sources and self.sources.index('sfmr') != 0:
            tmp = self.sources[0]
            des_idx = self.sources.index('sfmr')
            self.sources[0] = 'sfmr'
            self.sources[des_idx] = tmp

        if 'sfmr' in self.sources and self.sources.index('sfmr') != 0:
            self.logger.error((f"""Must put 'sfmr' in the head of """
                               f"""'sources' list"""))
            exit()

        if 'sfmr' in self.sources:
            if len(self.sources) != 2:
                self.logger.error((
                    f"""Must input 2 sources """
                    f"""when including SFMR"""))
                exit()
        else:
            if len(self.sources) < 1:
                self.logger.error((
                    f"""At least must input 1 sources """
                    f"""when not including SFMR"""))
                exit()

        self.sources_titles = self.CONFIG['plot'][
            'data_sources_title']

        self.sources_str = self.gen_sources_str()

        if 'sfmr' in self.sources:
            self.load_match_data_sources()
        # {'TC_name': [], 'date_time': []}

        self.compare_sources()

        if 'sfmr' in self.sources:
            self.update_match_data_sources()

    def load_match_data_sources(self):
        match_str = f'sfmr_vs_{self.sources[1]}'
        dir = (self.CONFIG['result']['dirs']['statistic'][
            'match_of_data_sources']
               + f'{match_str}/{self.basin}/')
        file_path = f'{dir}{self.basin}_match_{match_str}.pkl'
        self.match_data_source_file_path = file_path

        if not os.path.exists(file_path):
            self.match_data_sources = pd.DataFrame(columns=[
                'TC_sid', 'datetime', 'match'])
        else:
            with open(file_path, 'rb') as f:
                self.match_data_sources = pickle.load(f)

    def update_match_data_sources(self):
        if os.path.exists(self.match_data_source_file_path):
            os.remove(self.match_data_source_file_path)
        os.makedirs(os.path.dirname(self.match_data_source_file_path),
                    exist_ok=True)
        self.match_data_sources.to_pickle(
            self.match_data_source_file_path)

    def gen_sources_str(self):
        sources_str = ''
        for idx, src in enumerate(self.sources):
            if idx < len(self.sources) - 1:
                sources_str = f"""{sources_str}{src.upper()} and """
            else:
                sources_str = f"""{sources_str}{src.upper()}"""

        return sources_str

    def compare_sources(self):
        self.logger.info((
            f"""Comparing wind speed from different sources"""))
        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name'][
            self.basin]
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)
        if ('smap' in self.sources
            and 'smap_prediction' in self.sources):
            # There are matched SMAP data that covers the center of TC
            # We want to use them to compare with SMAP prediction
            # So we need to get corrseponding TC's sid and datetime
            years = [y for y in range(self.period[0].year,
                                      self.period[1].year + 1)]
            cover_center_sid = []
            cover_center_hour = []
            for y in years:
                table_name = utils.gen_tc_satel_era5_tablename(
                    'smap', y, self.basin)
                MatchTable = utils.get_class_by_tablename(self.engine,
                                                          table_name)
                if MatchTable is None:
                    continue
                # Query for sid and datetime of TCs of which centers
                # are covered by SMAP
                cover_center_query = self.session.query(
                    MatchTable).filter(MatchTable.x == 0,
                                       MatchTable.y == 0)
                for row in cover_center_query:
                    cover_center_sid.append(row.sid)
                    cover_center_hour.append(row.era5_datetime)

        tc_query = self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        )
        total = tc_query.count()

        windspd_bias_frames = []
        # Filter TCs during period
        for idx, tc in enumerate(tc_query):
            # if 'sfmr' in self.sources:
            #     if tc.wind < 64:
            #         continue
            if ('smap' in self.sources
                and 'smap_prediction' in self.sources):
                # 
                if tc.sid not in cover_center_sid:
                    continue
                sid_index = cover_center_sid.index(tc.sid)
                interp_dt = cover_center_hour[sid_index]
                # When IBTrACS datetime does not equal to the datetime
                # of interpolated TC of which center covered by SMAP
                if tc.date_time != interp_dt:
                    if idx == total - 1:
                        break
                    next_tc = tc_query[idx + 1]
                    # This TC and next TC are not the same
                    if tc.sid != next_tc.sid:
                        continue
                    # Datetime of interpolated TC not between this TC
                    # and next TC
                    if not (tc.date_time < interp_dt
                            and next_tc.date_time > interp_dt):
                        continue
                    # Interpolate a new TC record
                    hour_shift = int(
                        (interp_dt - tc.date_time).seconds / 3600)
                    interp_tc = self.interp_tc(hour_shift, tc,
                                               next_tc)
                    del tc
                    tc = interp_tc

            # if tc.wind < 64:
            #     continue
            # if tc.lat < 10 or tc.lat > 20:
            #     continue
            # if tc.lon < 110 or tc.lon > 120:
            #     continue
            # if tc.r34_ne is None:
            #     continue
            converted_lon = utils.longitude_converter(
                tc.lon, '360', '-180')
            if bool(globe.is_land(tc.lat, converted_lon)):
                continue
            # Draw windspd from different sources
            success = False
            if 'sfmr' not in self.sources:
                success, need_exit = self.compare_with_one_tc_record(tc)
                if success:
                    print((f"""Comparing {self.sources_str} with """
                           f"""IBTrACS record of TC {tc.name} on """
                           f"""{tc.date_time}"""))
                else:
                    print((f"""Skiping comparsion of """
                           f"""{self.sources_str} with """
                           f"""IBTrACS record of TC {tc.name} """
                           f"""on {tc.date_time}"""))
            else:
                if idx < total - 1:
                    next_tc = tc_query[idx + 1]
                    if tc.sid == next_tc.sid:
                        success, windspd_bias_df_between_two_tcs = \
                                self.compare_with_sfmr(tc, next_tc)
                        if success:
                            windspd_bias_frames.append(
                                windspd_bias_df_between_two_tcs)

        if 'sfmr' in self.sources and len(windspd_bias_frames):
            try:
                all_windspd_bias_to_sfmr = pd.concat(
                    windspd_bias_frames).reset_index(drop=True)

                save_root_dir = self.CONFIG['result']['dirs'][
                    'statistic']['windspd_bias_to_sfmr']
                compare_dir = f'{self.sources[1]}_vs_sfmr'

                save_name = f'{self.basin}'
                for dt in self.period:
                    save_name = (f"""{save_name}_"""
                                 f"""{dt.strftime('%Y%m%d%H%M%S')}""")

                save_dir = (f"""{save_root_dir}{compare_dir}/"""
                            f"""{save_name}/""")
                os.makedirs(save_dir, exist_ok=True)

                all_windspd_bias_to_sfmr.to_pickle(
                    f'{save_dir}{save_name}.pkl')
            except Exception as msg:
                breakpoint()
                exit(msg)

        print('Done')

    def update_no_match_between_tcs(self, hours, tc, next_tc):
        """Reocrd the nonexistence of matchup of data sources.

        """
        tc_sid_list = []
        tc_dt_list = []
        match_list = []

        for h in range(hours):
            interp_tc = self.interp_tc(h, tc, next_tc)
            tc_sid_list.append(interp_tc.sid)
            tc_dt_list.append(interp_tc.date_time)
            match_list.append(False)

        additional_match_of_data_sources_df = pd.DataFrame(
            {'TC_sid': tc_sid_list,
             'datetime': tc_dt_list,
             'match': match_list,
            }
        )
        self.match_data_sources = self.match_data_sources.\
                append(additional_match_of_data_sources_df)
        self.match_data_sources.sort_values(
            by=['datetime'], inplace=True, ignore_index=True)
        self.match_data_sources.drop_duplicates(
                inplace=True, ignore_index=True)

    def update_one_row_of_match(self, interp_tc, match):
        one_row_df = pd.DataFrame(
            {'TC_sid': [interp_tc.sid],
             'datetime': [interp_tc.date_time],
             'match': [match],
            }
        )
        self.match_data_sources = self.match_data_sources.\
                append(one_row_df)
        self.match_data_sources.sort_values(
            by=['datetime'], inplace=True, ignore_index=True)
        self.match_data_sources.drop_duplicates(
            inplace=True, ignore_index=True)

    def compare_with_sfmr(self, tc, next_tc):
        windspd_bias_df_between_two_tcs = None
        # if (next_tc.date_time < self.match_data_sources['date_time'][-1]
        #     or tc.date_time > self.match_data_sources['date_time'][0]):
        #     # Check if match of data sources exist during the life
        #     # period of TC
        #     if tc.name not in self.match_data_sources[
        #         'TC_name'].unique():
        #         return False, windspd_bias_df_between_two_tcs

        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are far away in time
        if delta.days:
            return False, windspd_bias_df_between_two_tcs
        hours = int(delta.seconds / 3600)
        # Time interval between two TCs are less than one hour
        if not hours:
            return False, windspd_bias_df_between_two_tcs

        hit_dt = []
        match_dt = []
        for h in range(hours):
            interp_tc = self.interp_tc(h, tc, next_tc)
            same_dt_df = self.match_data_sources.loc[
                self.match_data_sources['datetime'] == \
                interp_tc.date_time]
            if not len(same_dt_df):
                continue
            same_sid_dt_df = same_dt_df.loc[
                same_dt_df['TC_sid'] == interp_tc.sid]
            if not len(same_sid_dt_df):
                continue
            if len(same_sid_dt_df) == 1:
                hit_dt.append(interp_tc.date_time)
                if same_sid_dt_df['match'].iloc[0]:
                    match_dt.append(interp_tc.date_time)
            else:
                print(same_sid_dt_df)
                self.logger.error((f"""Strange: two or more """
                                   f"""comparison has same sid """
                                   f"""and datetime"""))
                exit()

        hit_count = len(hit_dt)
        match_count = len(match_dt)
        if hit_count == hours and not match_count:
            print((f"""[Skip] All internal hours of TC """
                   f"""{tc.name} between {tc.date_time} """
                   f"""and {next_tc.date_time}"""))
            return False, windspd_bias_df_between_two_tcs

        # Check existence of SFMR between two IBTrACS records
        existence, spatial_temporal_info = self.sfmr_exists(
            tc, next_tc)
        if not existence:
            # First executed here between particular two TCs
            if hit_count < hours:
                # update match of data sources
                self.update_no_match_between_tcs(hours, tc, next_tc)

            print((f"""[Not exist] SFMR of TC {tc.name} between """
                   f"""{tc.date_time} and {next_tc.date_time}"""))
            return False, windspd_bias_df_between_two_tcs

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
        hour_info_pt_idx = self.sfmr_rounded_hours(
            tc, next_tc, spatial_temporal_info)
        if not len(hour_info_pt_idx):
            # First executed here between particular two TCs
            if hit_count < hours:
                # update match of data sources
                self.update_no_match_between_tcs(hours, tc, next_tc)

            print((f"""[Fail rounding to hour] SFMR of TC {tc.name} """
                   f"""between {tc.date_time} and """
                   f"""{next_tc.date_time}"""))
            return False, windspd_bias_df_between_two_tcs

        if hit_count == hours:
            # In this condition, `match count` is larger than zero.  
            # Just compare the datetimes when `match` value is True
            # and no need to update `match_data_sources`
            success, windspd_bias_df_between_two_tcs = \
                    self.compare_with_sfmr_all_records_hit(
                        tc, next_tc, hours, match_dt,
                        spatial_temporal_info, hour_info_pt_idx)
        else:
            # First executed here between particular two TCs.  
            # For each hour, compare SFMR wind speed with wind speed
            # from other sources and have to update `match_data_sources`
            success, windspd_bias_df_between_two_tcs = \
                    self.compare_with_sfmr_not_all_records_hit(
                        tc, next_tc, hours, spatial_temporal_info, 
                        hour_info_pt_idx)

        return success, windspd_bias_df_between_two_tcs

    def compare_with_sfmr_not_all_records_hit(self, tc, next_tc, hours,
                                              spatial_temporal_info, 
                                              hour_info_pt_idx):
        windspd_bias_frames = []
        tc_sid_list = []
        tc_dt_list = []
        match_list = []
        need_exit = False

        for h in range(hours):
            interp_tc = self.interp_tc(h, tc, next_tc)
            if interp_tc.date_time not in hour_info_pt_idx.keys():
                # update corrseponding match
                self.update_one_row_of_match(interp_tc, False)
                print((f"""[Not exist] SFMR of TC {tc.name} near """
                       f"""{interp_tc.date_time}"""))
                continue

            success, hourly_windspd_bias_df = \
                    self.compare_with_sfmr_around_interped_tc(
                        spatial_temporal_info,
                        hour_info_pt_idx[interp_tc.date_time],
                        interp_tc)
            tc_sid_list.append(interp_tc.sid)
            tc_dt_list.append(interp_tc.date_time)

            if success:
                match_list.append(True)
                windspd_bias_frames.append(hourly_windspd_bias_df)

                print((f"""[Match] {self.sources_str} """
                       f"""of TC {interp_tc.name} """
                       f"""on {interp_tc.date_time}"""))
            else:
                if hourly_windspd_bias_df is None:
                    # Normal fail, need continue comparing
                    match_list.append(False)
                    print((f"""[Not match] {self.sources_str} """
                           f"""of TC {interp_tc.name} """
                           f"""on {interp_tc.date_time}"""))
                elif hourly_windspd_bias_df == 'exit':
                    # Exception occurs, need save matchup
                    # of data sources before shutdowning
                    # program
                    need_exit = True
                    tc_sid_list.pop()
                    tc_dt_list.pop()
                    break

            # if not success:
            #     return False, windspd_bias_df_between_two_tcs

        additional_match_of_data_sources_df = pd.DataFrame(
            {'TC_sid': tc_sid_list,
             'datetime': tc_dt_list,
             'match': match_list,
            }
        )
        self.match_data_sources = self.match_data_sources.append(
            additional_match_of_data_sources_df)
        self.match_data_sources.sort_values(by=['datetime'],
                                            inplace=True,
                                            ignore_index=True)
        self.match_data_sources.drop_duplicates(inplace=True,
                                                ignore_index=True)

        if need_exit:
            self.update_match_data_sources()
            exit(1)

        if not len(windspd_bias_frames):
            return False, None

        windspd_bias_df_between_two_tcs = pd.concat(windspd_bias_frames)

        return True, windspd_bias_df_between_two_tcs

    def compare_with_sfmr_all_records_hit(
        self, tc, next_tc, hours, match_dt, spatial_temporal_info,
        hour_info_pt_idx):
        #
        windspd_bias_frames = []
        for h in range(hours):
            interp_tc = self.interp_tc(h, tc, next_tc)
            if interp_tc.date_time not in match_dt:
                print((f"""[Skip] {self.sources_str} """
                       f"""of TC {interp_tc.name} """
                       f"""on {interp_tc.date_time}"""))
                continue

            success, hourly_windspd_bias_df = \
                    self.compare_with_sfmr_around_interped_tc(
                        spatial_temporal_info,
                        hour_info_pt_idx[interp_tc.date_time],
                        interp_tc)
            print((f"""[Redo] {self.sources_str} """
                   f"""of TC {interp_tc.name} """
                   f"""on {interp_tc.date_time}"""))
            if not success:
                self.logger.error('Match True but not success')
                breakpoint()
                exit()

            windspd_bias_frames.append(hourly_windspd_bias_df)

        if not len(windspd_bias_frames):
            return False, None

        windspd_bias_df_between_two_tcs = pd.concat(windspd_bias_frames)

        return True, windspd_bias_df_between_two_tcs

    def compare_with_sfmr_around_interped_tc(
        self, sfmr_brief_info, one_hour_info_pt_idx, interp_tc):
        # Reference function `compare_with_one_tc_record`
        # Because we need to overlay
        # SFMR tracks and averaged SFMR wind speed
        # onto other sources' wind speed map, the inputted subplots
        # number should minus one (SFMR source)
        subplots_row, subplots_col, fig_size = \
                utils.get_subplots_row_col_and_fig_size(
                    len(self.sources) - 1)
        if subplots_row * subplots_col > 1:
            text_subplots_serial_number = True
        else:
            text_subplots_serial_number = False
        fig, axs = plt.subplots(subplots_row, subplots_col,
                                figsize=fig_size, sharey=True)
        if len(self.sources) > 2:
            axs_list = []
            for idx, src in enumerate(self.sources):
                if src == 'sfmr':
                    continue
                if subplots_row == 1:
                    col_idx = (idx - 1) % subplots_col
                    ax = axs[col_idx]
                elif subplots_row > 1:
                    row_idx = int((idx - 1) / subplots_col)
                    col_idx = (idx - 1) % subplots_col
                    ax = axs[row_idx][col_idx]

                axs_list.append(ax)
        else:
            axs_list = [axs]

        success, era5_lons, era5_lats = \
                self.get_smap_prediction_lonlat(interp_tc)
        if not success:
            return False, None
        del success
        # North, South, West, East
        draw_region = [min(era5_lats), max(era5_lats),
                       min(era5_lons), max(era5_lons)]

        lons = dict()
        lats = dict()
        windspd = dict()
        radii_area = dict()
        mesh = dict()
        diff_mins = dict()

        max_windspd = -1
        for src in self.sources:
            if src == 'sfmr':
                success, sfmr_tracks, sfmr_dts, sfmr_lons, \
                        sfmr_lats, sfmr_windspd = \
                        self.get_sfmr_windspd_along_track(
                            interp_tc, era5_lons, era5_lats,
                            sfmr_brief_info, one_hour_info_pt_idx)
                if not success:
                    return False, None
                for single_track_windspd in sfmr_windspd:
                    if max(single_track_windspd) > max_windspd:
                        max_windspd = max(single_track_windspd)
            else:
                try:
                    # Get lons, lats, windspd
                    success, lons[src], lats[src], windspd[src], \
                            mesh[src], diff_mins[src] = \
                            self.get_sources_xyz_matrix(
                                src, interp_tc, era5_lons, era5_lats,
                                sfmr_brief_info, one_hour_info_pt_idx)
                    if not success:
                        if lons[src] is None:
                            # Normal fail, need continue comparing
                            return False, None
                        elif lons[src] == 'exit':
                            # Exception occurs, need save matchup
                            # of data sources before shutdowning
                            # program
                            return False, 'exit'
                    if src == 'smap':
                        # Not all points of area has this feature
                        # Make sure it is interpolated properly
                        diff_mins[src] = utils.\
                                interp_satel_era5_diff_mins_matrix(
                                    diff_mins[src])
                    # Get max windspd
                    if windspd[src].max() > max_windspd:
                        max_windspd = windspd[src].max()
                except Exception as msg:
                    breakpoint()
                    exit(msg)

        # To quantificationally validate simulated SMAP wind,
        # we need to extract SMAP points matching SFMR points
        windspd_bias_df_list = []
        for src in self.sources:
            if src == 'sfmr':
                continue
            tmp_df = utils.validate_with_sfmr(
                src, interp_tc.date_time, sfmr_dts, sfmr_lons,
                sfmr_lats, sfmr_windspd, lons[src], lats[src],
                windspd[src], mesh[src], diff_mins[src])
            windspd_bias_df_list.append(tmp_df)

        if not len(windspd_bias_df_list):
            return False, None

        windspd_bias_df = pd.concat(windspd_bias_df_list)

        index = 0
        tc_dt = interp_tc.date_time
        subplot_title_suffix = (
            f"""{tc_dt.strftime('%H%M UTC %d')} """
            f"""{calendar.month_name[tc_dt.month]} """
            f"""{tc_dt.year} {interp_tc.name}"""
        )
        # Draw windspd
        for src in self.sources:
            try:
                if src == 'sfmr':
                    continue
                ax = axs_list[index]

                utils.draw_windspd(self, fig, ax, interp_tc.date_time,
                                   lons[src], lats[src], windspd[src],
                                   max_windspd, mesh[src], custom=True,
                                   region=draw_region)
                utils.draw_sfmr_windspd_and_track(
                    self, fig, ax, interp_tc.date_time, sfmr_tracks,
                    sfmr_lons, sfmr_lats, sfmr_windspd, max_windspd)
                # if text_subplots_serial_number:
                #     ax.text(-0.1, 1.025,
                #             f'({string.ascii_lowercase[index]})',
                #             transform=ax.transAxes, fontsize=16,
                #             fontweight='bold', va='top', ha='right')
                ax.title.set_text((f"""{self.sources_titles[src]}"""
                                   f""" {subplot_title_suffix}"""))
                ax.title.set_size(10)
                index += 1
            except Exception as msg:
                breakpoint()
                exit(msg)

        fig.subplots_adjust(wspace=0.4)

        dt_str = interp_tc.date_time.strftime('%Y_%m%d_%H%M')
        fig_dir = self.CONFIG['result']['dirs']['fig']['root']
        for idx, src in enumerate(self.sources):
            if not idx:
                fig_dir = f"""{fig_dir}{src}_vs"""
            elif idx < len(self.sources) - 1:
                fig_dir = f"""{fig_dir}_{src}_vs"""
            else:
                fig_dir = f"""{fig_dir}_{src}/"""

        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'{dt_str}_{interp_tc.name}.png'
        plt.savefig(f'{fig_dir}{fig_name}')
        plt.clf()

        return True, windspd_bias_df

    def sfmr_rounded_hours(self, tc, next_tc, spatial_temporal_info):
        # Include start hour, but not end hour
        # To let all intervals same
        hours_between_two_tcs = []
        datetime_area = dict()
        hour_info_pt_idx = dict()

        delta = next_tc.date_time - tc.date_time
        hours = int(delta.seconds / 3600)

        # if (next_tc.date_time == datetime.datetime(2018, 7, 10, 18, 0)
        #     or tc.date_time == datetime.datetime(2018, 7, 10, 18, 0)):
        #     breakpoint()

        for h in range(hours):
            interp_dt = tc.date_time + datetime.timedelta(
                seconds = h * 3600)
            hours_between_two_tcs.append(interp_dt)

            datetime_area[interp_dt] = dict()

            tc_pre = tc
            interp_tc = self.interp_tc(h, tc, next_tc)
            tc_aft = tc
            if tc.date_time == next_tc.date_time:
            # if tc_pre.date_time != tc_aft.date_time:
                breakpoint()

            datetime_area[interp_dt]['lon1'] = \
                    interp_tc.lon - self.half_reg_edge
            datetime_area[interp_dt]['lon2'] = \
                    interp_tc.lon + self.half_reg_edge
            datetime_area[interp_dt]['lat1'] = \
                    interp_tc.lat - self.half_reg_edge
            datetime_area[interp_dt]['lat2'] = \
                    interp_tc.lat + self.half_reg_edge

        # traverse all brief info of SFMR file
        for info_idx, info in enumerate(spatial_temporal_info):
            year = info.start_datetime.year
            file_path = (
                f"""{self.CONFIG['sfmr']['dirs']['hurr']}"""
                f"""{year}/{info.hurr_name}/{info.filename}"""
            )
            dataset = netCDF4.Dataset(file_path)

            # VERY VERY IMPORTANT: netCDF4 auto mask may cause problems,
            # so must disable auto mask
            dataset.set_auto_mask(False)
            vars = dataset.variables
            length = len(vars['TIME'])

            # Traverse all data points of selected SFMR file
            for i in range(length):
                # Round SFMR data point's datetime to hours
                try:
                    pt_date = vars['DATE'][i]
                    pt_time = vars['TIME'][i]
                    # It seems that near the end of SFMR data array,
                    # DATE will be 0
                    if pt_date == 0:
                        continue

                    pt_datetime = datetime.datetime.combine(
                        utils.sfmr_nc_converter('DATE', pt_date),
                        utils.sfmr_nc_converter('TIME', pt_time)
                    )
                    rounded_hour = utils.hour_rounder(pt_datetime)
                except Exception as msg:
                    breakpoint()
                    exit(msg)

                # Check whether rounded hours are in hours between
                # two TCs
                if rounded_hour not in hours_between_two_tcs:
                    continue

                lon = (vars['LON'][i] + 360) % 360
                lat = vars['LAT'][i]

                # Check whether SFMR data points are in area around
                # TC at rounded hour
                if (lon < datetime_area[rounded_hour]['lon1']
                    or lon > datetime_area[rounded_hour]['lon2']
                    or lat < datetime_area[rounded_hour]['lat1']
                    or lat > datetime_area[rounded_hour]['lat2']):
                    continue
                rounded_hour_idx = hours_between_two_tcs.index(
                    rounded_hour)

                # Add SFMR data point index into `hour_info_pt_idx`
                if rounded_hour not in hour_info_pt_idx:
                    hour_info_pt_idx[rounded_hour] = dict()
                if info_idx not in hour_info_pt_idx[rounded_hour]:
                    hour_info_pt_idx[rounded_hour][info_idx] = []

                hour_info_pt_idx[rounded_hour][info_idx].append(i)

        return hour_info_pt_idx

    def sfmr_exists(self, tc, next_tc):
        """Check the existence of SFMR data between two
        temporally neighbouring IBTrACS records of a same TC
        and get the brief info of these SFMR data.

        Parameters
        ----------
        tc : object describing a row of IBTrACS table
            An IBTrACS TC record eariler.
        next_tc : object describing a row of IBTrACS table
            Another IBTrACS record of the same TC later.

        Returns
        -------
        bool
            True if SFMR data exists, False otherwise.
        spatial_temporal_sfmr_info: object describing rows of the \
                brief info table of all SFMR data
            Brief info of SFMR data which spatially and temporally \
                    between two IBTrACS records.

        """

        # Rough temporally check
        temporal_existence, temporal_sfmr_info = \
                self.sfmr_temporally_exists(tc, next_tc)
        if not temporal_existence:
            return False, None

        # Rough spaitally check
        spatial_existence, spatial_temporal_sfmr_info = \
                self.sfmr_spatially_exists(tc, next_tc,
                                           temporal_sfmr_info)
        if not spatial_existence:
            return False, None

        # Detailed check
        # ???

        return True, spatial_temporal_sfmr_info

    def sfmr_temporally_exists(self, tc, next_tc):
        existence = False
        temporal_info = []

        table_name = self.CONFIG['sfmr']['table_names']['brief_info']
        BriefInfo = utils.get_class_by_tablename(self.engine,
                                                 table_name)

        direct_query = self.session.query(BriefInfo).filter(
            BriefInfo.end_datetime > tc.date_time,
            BriefInfo.start_datetime < next_tc.date_time)
        count_sum = direct_query.count()

        if count_sum:
            existence = True
            for row in direct_query:
                temporal_info.append(row)

        return existence, temporal_info

    def sfmr_spatially_exists(self, tc, next_tc, temporal_info):
        # It seems that need to compare rectangle of SFMR range with
        # regression range of area around TC in a specified hour, not
        # the period between two neighbouring TCs
        existence = False
        spatial_temporal_info = []

        delta = next_tc.date_time - tc.date_time

        # Calculate the circumscribed rectangle of all area of regression
        # on every hour between two neighbouring TCs
        hours = int(delta.seconds / 3600)
        # Spatial shift
        try:
            lon_shift, lat_shift = utils.get_center_shift_of_two_tcs(
                next_tc, tc)
            hourly_lon_shift = lon_shift / hours
            hourly_lat_shift = lat_shift / hours
        except Exception as msg:
            breakpoint()
            exit(msg)
        half_reg_edge = self.CONFIG['regression']['edge_in_degree'] / 2
        corners = {'left': [], 'top': [], 'right': [], 'bottom': []}
        # Extract from the interval between two TC records
        for h in range(hours):
            interp_tc_lon = (h * hourly_lon_shift + tc.lon)
            interp_tc_lat = (h * hourly_lat_shift + tc.lat)
            corners['left'].append(interp_tc_lon - half_reg_edge)
            corners['top'].append(interp_tc_lat + half_reg_edge)
            corners['right'].append(interp_tc_lon + half_reg_edge)
            corners['bottom'].append(interp_tc_lat - half_reg_edge)
        # Describe rectangle of regression area between two TCs
        left_top_tc = utils.Point(min(corners['left']),
                                  max(corners['top']))
        right_bottom_tc = utils.Point(max(corners['right']),
                                      min(corners['bottom']))

        for info in temporal_info:
            left_top_sfmr = utils.Point(info.min_lon, info.max_lat)
            right_bottom_sfmr = utils.Point(info.max_lon, info.min_lat)
            if utils.doOverlap(left_top_tc, right_bottom_tc,
                               left_top_sfmr, right_bottom_sfmr):
                existence = True
                spatial_temporal_info.append(info)

        return existence, spatial_temporal_info

    def interp_tc(self, h, tc, next_tc):
        """Get sid, interpolated datetime, longitude and latitude of
        two neighbouring TC records.

        """
        try:
            # Temporal shift
            delta = next_tc.date_time - tc.date_time
            hours = int(delta.seconds / 3600)
            # Spatial shift
            lon_shift, lat_shift = utils.get_center_shift_of_two_tcs(
                next_tc, tc)
            hourly_lon_shift = lon_shift / hours
            hourly_lat_shift = lat_shift / hours

            # Get IBTrACS table
            table_name = self.CONFIG['ibtracs']['table_name'][
                self.basin]
            IBTrACS = utils.get_class_by_tablename(self.engine,
                                                   table_name)
            # ATTENTIONL: DO NOT direct use `interp_tc = tc`
            # Because it makes a link between two variables
            # any modification will simultaneously change two variables
            interp_tc = IBTrACS()
            interp_tc.sid = tc.sid
            interp_tc.name = tc.name
            # interp_tc.basin = tc.basin
            # interp_tc.pres = tc.pres
            # interp_tc.wind = tc.wind
            # interp_tc.r34_ne = tc.r34_ne
            # interp_tc.r34_se = tc.r34_se
            # interp_tc.r34_sw = tc.r34_sw
            # interp_tc.r34_nw = tc.r34_nw
            # interp_tc.r50_ne = tc.r50_ne
            # interp_tc.r50_ne = tc.r50_ne
            # interp_tc.r50_se = tc.r50_se
            # interp_tc.r50_sw = tc.r50_sw
            # interp_tc.r64_nw = tc.r64_nw
            # interp_tc.r64_se = tc.r64_se
            # interp_tc.r64_sw = tc.r64_sw
            # interp_tc.r64_nw = tc.r64_nw
            # Only interpolate `date_time`, `lon`, `lat` variables
            # Other variables stays same with `tc`
            interp_tc.date_time = tc.date_time + datetime.timedelta(
                seconds = h * 3600)
            interp_tc.lon = (h * hourly_lon_shift + tc.lon)
            interp_tc.lat = (h * hourly_lat_shift + tc.lat)
        except Exception as msg:
            breakpoint()
            exit(msg)

        return interp_tc

    def compare_with_one_tc_record(self, tc):
        """

        Returns
        -------
        success: bool
            Success comparing or not.
        need_exit: bool
            Whether exception occurs and need exit.

        """
        subplots_row, subplots_col, fig_size = \
                utils.get_subplots_row_col_and_fig_size(len(
                    self.sources))
        if subplots_row * subplots_col > 1:
            text_subplots_serial_number = True
        else:
            text_subplots_serial_number = False

        fig, axs = plt.subplots(subplots_row, subplots_col,
                                figsize=fig_size, sharey=True)
        axs_list = []
        for idx, src in enumerate(self.sources):
            if subplots_row == 1:
                col_idx = idx % subplots_col
                ax = axs[col_idx]
            elif subplots_row > 1:
                row_idx = int(idx / subplots_col)
                col_idx = idx % subplots_col
                ax = axs[row_idx][col_idx]

            axs_list.append(ax)

        success, era5_lons, era5_lats = \
                self.get_smap_prediction_lonlat(tc)
        if not success:
            return False, False
        del success
        # South, North, West, East
        draw_region = [min(era5_lats), max(era5_lats),
                       min(era5_lons), max(era5_lons)]

        lons = dict()
        lats = dict()
        windspd = dict()
        radii_area = dict()
        mesh = dict()
        diff_mins = dict()

        max_windspd = -1
        for idx, src in enumerate(self.sources):
            ax = axs_list[idx]

            # Get lons, lats, windspd
            success, lons[src], lats[src], windspd[src], mesh[src], \
                    diff_mins[src] = self.get_sources_xyz_matrix(
                        src, tc, era5_lons, era5_lats)
            if not success:
                if lons[src] is None:
                    # Normal fail, need continue comparing
                    return False, False
                elif lons[src] == 'exit':
                    # Exception occurs, need save matchup
                    # of data sources before shutdowning
                    # program
                    return False, True
            # Get max windspd
            if windspd[src].max() > max_windspd:
                max_windspd = windspd[src].max()
            # Draw wind radii
            radii_area[src] = utils.draw_ibtracs_radii(ax, tc,
                                                       self.zorders)

        tc_dt = tc.date_time
        subplot_title_suffix = (
            f"""{tc_dt.strftime('%H%M UTC %d')} """
            f"""{calendar.month_name[tc_dt.month]} """
            f"""{tc_dt.year} {tc.name}"""
        )
        # Draw windspd
        for idx, src in enumerate(self.sources):
            try:
                ax = axs_list[idx]

                utils.draw_windspd(self, fig, ax, tc.date_time,
                                   lons[src], lats[src], windspd[src],
                                   max_windspd, mesh[src], custom=True,
                                   region=draw_region)
                # if text_subplots_serial_number:
                #     ax.text(-0.1, 1.025, f'({string.ascii_lowercase[idx]})',
                #             transform=ax.transAxes, fontsize=16,
                #             fontweight='bold', va='top', ha='right')
                ax.title.set_text((f"""{self.sources_titles[src]}"""
                                   f""" {subplot_title_suffix}"""))
                ax.title.set_size(10)
            except Exception as msg:
                breakpoint()
                exit(msg)

        fig.subplots_adjust(wspace=0.4)

        dt_str = tc.date_time.strftime('%Y_%m%d_%H%M')
        fig_dir = (f"""{self.CONFIG['result']['dirs']['fig']['root']}"""
                   f"""ibtracs_vs""")
        for idx, src in enumerate(self.sources):
            if idx < len(self.sources) - 1:
                fig_dir = f"""{fig_dir}_{src}_and"""
            else:
                fig_dir = f"""{fig_dir}_{src}/"""

        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'{dt_str}_{tc.name}.png'
        plt.savefig(f'{fig_dir}{fig_name}')
        plt.clf()

        return True, False

    def get_sources_xyz_matrix(self, src, tc, era5_lons=None,
                               era5_lats=None, pt_lons=None,
                               pt_lats=None, pt_dts=None,
                               sfmr_brief_info=None,
                               one_hour_info_pt_idx=None):
        if src == 'ccmp':
            return self.get_ccmp_xyz_matrix(tc)
        elif src == 'era5':
            return self.get_era5_xyz_matrix(tc, era5_lons, era5_lats)
        elif src == 'smap':
            return self.get_smap_xyz_matrix(tc, era5_lons, era5_lats)
        elif src == 'smap_prediction':
            if ('smap' in self.sources or 'era5' in self.sources
                or 'sfmr' in self.sources):
                return self.get_smap_prediction_pts_or_xyz_matrix(
                    'matrix', tc, era5_lons, era5_lats, pt_lons, pt_lats,
                    pt_dts)
            # elif 'sfmr' in self.sources:
            #     return self.get_smap_prediction_pts_or_xyz_matrix(
            #         'points', tc, era5_lons, era5_lats, pt_lons, pt_lats,
            #         pt_dts)

    def get_sfmr_windspd_along_track(self, tc, grid_lons, grid_lats,
                                     sfmr_brief_info,
                                     one_hour_info_pt_idx):
        all_tracks = []
        all_dts = []
        all_lons = []
        all_lats = []
        all_windspd = []

        # Logger information
        # self.logger.info(f'Getting xyz_matrix of SFMR around TC')

        root_dir = self.CONFIG['sfmr']['dirs']['hurr']

        # Get SFMR windspd
        for info_idx in one_hour_info_pt_idx.keys():
            data_indices = one_hour_info_pt_idx[info_idx]
            brief_info = sfmr_brief_info[info_idx]
            file_dir = (
                f"""{root_dir}{brief_info.start_datetime.year}/"""
                f"""{brief_info.hurr_name}/""")
            file_path = f'{file_dir}{brief_info.filename}'
            # Firstly try first-come-first-count method
            # Secondly try square-average method
            try:
                result = utils.get_sfmr_track_and_windspd(file_path,
                                                          data_indices)
                if result[0] is None:
                    return False, None, None, None, None, None
                all_tracks.append(result[0])
                all_dts.append(result[1])
                all_lons.append(result[2])
                all_lats.append(result[3])
                all_windspd.append(result[4])
            except Exception as msg:
                breakpoint()
                exit(msg)

        # For our verification, we do not use SFMR observations whose
        # wind speed is below 15 m/s, as the singal-to-noise ration in
        # the SFMR measurement becomes unfavorable at lower wind speeds.

        # Meissner, Thomas, Lucrezia Ricciardulli, and Frank J. Wentz.
        # “Capability of the SMAP Mission to Measure Ocean Surface Winds
        # in Storms.” Bulletin of the American Meteorological Society 98,
        # no. 8 (March 7, 2017): 1660–77.
        # https://doi.org/10.1175/BAMS-D-16-0052.1.

        # J. Carswell 2015, personal communication
        final_dts = []
        final_lons = []
        final_lats = []
        final_windspd = []
        try:
            for track_idx, single_track_windspd in enumerate(
                all_windspd):
                #
                tmp_dts = []
                tmp_lons = []
                tmp_lats = []
                tmp_windspd = []
                for pt_idx, pt_windspd in enumerate(
                    single_track_windspd):
                    #
                    if pt_windspd >= 15:
                        tmp_dts.append(all_dts[track_idx][pt_idx])
                        tmp_lons.append(all_lons[track_idx][pt_idx])
                        tmp_lats.append(all_lats[track_idx][pt_idx])
                        tmp_windspd.append(
                            all_windspd[track_idx][pt_idx])
                if len(tmp_windspd):
                    final_dts.append(tmp_dts)
                    final_lons.append(tmp_lons)
                    final_lats.append(tmp_lats)
                    final_windspd.append(tmp_windspd)

        except Exception as msg:
            breakpoint()
            exit(msg)

        if not len(final_windspd):
            return False, None, None, None, None, None
        else:
            return (True, all_tracks, final_dts, final_lons, final_lats,
                    final_windspd)

    def get_smap_prediction_pts_or_xyz_matrix(
        self, output, tc, era5_lons, era5_lats, pt_lons,
        pt_lats, pt_dts):
        """

        """
        try:
            # Test if era5 data can be extracted
            rounded_dt = utils.hour_rounder(tc.date_time)
            if rounded_dt.day != tc.date_time.day:
                return False, None, None, None, None, None

            # Create a new dataframe to store all points in region
            # Each point consists of ERA5 vars and SMAP windspd to
            # predict
            era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                            self.region,
                                            self.db_root_passwd,
                                            work=False,
                                            save_disk=self.save_disk,
                                            work_mode='',
                                            vars_mode='')
            # Get all useful environmental variables name
            col_names = []
            cols = utils.get_basic_satel_era5_columns(tc_info=True)
            era5_cols = era5_manager.get_era5_columns()
            cols = cols + era5_cols
            useless_cols_name = self.CONFIG['regression']\
                    ['useless_columns']['smap_era5']
            for col in cols:
                if col.name not in useless_cols_name:
                    col_names.append(col.name)

            smap_file_path = None

            # era5_lons, era5_lats = self.get_smap_prediction_lonlat(tc)
            # North, West, South, East,
            area = [max(era5_lats), min(era5_lons),
                    min(era5_lats), max(era5_lons)]
            if output == 'matrix':
                return self.get_smap_prediction_xyz_matrix(
                    tc, era5_lons, era5_lats, col_names, smap_file_path,
                    area)
            elif output == 'points':
                return self.get_smap_prediction_points(
                    tc, era5_lons, era5_lats, pt_lons, pt_lats, pt_dts,
                    col_names, area)
        except Exception as msg:
            self.logger.error((
                f"""function get_smap_prediction_pts_or_xyz_matrix: """
                f"""{msg}"""))
            return False, 'exit', None, None, None, None

    def get_smap_prediction_points(self, tc, era5_lons, era5_lats,
                                   pt_lons, pt_lats, pt_dts, col_names,
                                   area):
        try:
            if 'smap' in self.sources:
                self.logger.error((
                    f"""Need not to extract points when """
                    f"""comparing with SMAP"""))
                exit()
            elif 'sfmr' in self.sources:
                # Temporarily not modify diff_mins with SFMR data time
                env_data = self.get_env_data_of_points_with_coordinates(
                    tc, col_names, era5_lons, era5_lats, pt_lons,
                    pt_lats, pt_dts)
        except Exception as msg:
            breakpoint()
            exit(msg)

        # Get single levels vars of ERA5 around TC
        env_data, pres_lvls = self.add_era5_single_levels_data(
            env_data, col_names, tc, area)
        if env_data == 'exit':
            return False, 'exit', None, None, None, None

        # Get pressure levels vars of ERA5 around TC
        env_data = self.add_era5_pressure_levels_data(
            env_data, col_names, tc, area, pres_lvls)
        if env_data == 'exit':
            return False, 'exit', None, None, None, None

        # Predict SMAP windspd
        env_df = pd.DataFrame(env_data, columns=col_names,
                              dtype=np.float64)
        # Get original lon and lat before normalization
        # DO NOT use to_numpy() method, because it is a pointer to
        # DataFrame column, which changes with pointed column
        smap_windspd_xyz_matrix_dict = {
            'lon': list(env_df['lon']),
            'lat': list(env_df['lat']),
        }
        # Normalize data if the best model is trained after
        # normalization
        if 'normalization' in self.compare_instructions:
            scaler = MinMaxScaler()
            env_df[env_df.columns] = scaler.fit_transform(
                env_df[env_df.columns])

        model_dir = self.CONFIG['regression']['dirs']['tc']\
                ['lightgbm']['model']
        best_model = utils.load_best_model(model_dir, self.basin)
        smap_windspd_xyz_matrix_dict['windspd'] = list(
            best_model.predict(env_df))
        smap_windspd_xyz_matrix_df = pd.DataFrame(
            smap_windspd_xyz_matrix_dict, dtype=np.float64)
        # Pad SMAP windspd prediction around TC to all region
        smap_windspd = self.padding_tc_windspd_prediction(
            smap_windspd_xyz_matrix_df, era5_lons, era5_lats)

        # Return data
        return (True, era5_lons, era5_lats, smap_windspd,
                utils.if_mesh(era5_lons), None)

    def get_smap_prediction_xyz_matrix(self, tc, era5_lons, era5_lats,
                                       col_names, smap_file_path, area):
        # Only when not compare with SFMR but compare with SMAP
        if 'smap' in self.sources:
            satel_manager = satel_scs.SCSSatelManager(
                self.CONFIG, self.period, self.region,
                self.db_root_passwd, save_disk=self.save_disk,
                work=False)
            smap_file_path = satel_manager.download('smap',
                                                    tc.date_time)

            env_data = self.get_env_data_of_matrix_with_coordinates(
                tc, col_names, era5_lons, era5_lats, smap_file_path)
        # Only when not compare with SMAP but compare with SFMR
        elif 'sfmr' in self.sources:
            # Temporarily not modify diff_mins with SFMR data time
            env_data = self.get_env_data_of_matrix_with_coordinates(
                tc, col_names, era5_lons, era5_lats)

        if env_data is None:
            return False, None, None, None, None, None

        # Get single levels vars of ERA5 around TC
        env_data, pres_lvls = self.add_era5_single_levels_data(
            env_data, col_names, tc, area)
        if env_data == 'exit':
            return False, 'exit', None, None, None, None

        # Get pressure levels vars of ERA5 around TC
        env_data = self.add_era5_pressure_levels_data(
            env_data, col_names, tc, area, pres_lvls)
        if env_data == 'exit':
            return False, 'exit', None, None, None, None

        # Predict SMAP windspd
        env_df = pd.DataFrame(env_data, columns=col_names,
                              dtype=np.float64)
        # Get original lon and lat before normalization
        # DO NOT use to_numpy() method, because it is a pointer to
        # DataFrame column, which changes with pointed column
        smap_windspd_xyz_matrix_dict = {
            'lon': list(env_df['lon']),
            'lat': list(env_df['lat']),
        }
        # Normalize data if the best model is trained after
        # normalization
        if 'normalization' in self.compare_instructions:
            scaler = MinMaxScaler()
            env_df[env_df.columns] = scaler.fit_transform(
                env_df[env_df.columns])

        try:
            model_dir = self.CONFIG['regression']['dirs']['tc']\
                    ['lightgbm']['model']
            best_model = utils.load_best_model(model_dir, self.basin)
            smap_windspd_xyz_matrix_dict['windspd'] = list(
                best_model.predict(env_df))
            smap_windspd_xyz_matrix_df = pd.DataFrame(
                smap_windspd_xyz_matrix_dict, dtype=np.float64)
            # Pad SMAP windspd prediction around TC to all region
            smap_windspd = self.padding_tc_windspd_prediction(
                smap_windspd_xyz_matrix_df, era5_lons, era5_lats)
        except Exception as msg:
            breakpoint()
            exit(msg)

        # Return data
        return (True, era5_lons, era5_lats, smap_windspd,
                utils.if_mesh(era5_lons), None)

    def padding_tc_windspd_prediction(self, smap_windspd_xyz_matrix_df,
                                      era5_lons, era5_lats):
        df = smap_windspd_xyz_matrix_df
        windspd = np.full(shape=(len(era5_lats), len(era5_lons)),
                          fill_value=-1, dtype=float)

        for index, row in df.iterrows():
            try:
                lat = row['lat']
                lon = row['lon']
                lat_idx = era5_lats.index(lat)
                lon_idx = era5_lons.index(lon)
                windspd[lat_idx][lon_idx] = row['windspd']
            except Exception as msg:
                breakpoint()
                exit(msg)

        return windspd

    def get_env_data_of_points_with_coordinates(
        self, tc, col_names, era5_lons, era5_lats, pt_lons, pt_lats,
        pt_dts):
        """

        """
        env_data = []
        pts_num = len(pt_dts)

        for i in range(pts_num):
            row = dict()
            for name in col_names:
                row[name] = None

                row['lon'], lon_idx = \
                        utils.get_nearest_element_and_index(era5_lons,
                                                            pt_lons[i])
                row['lat'], lat_idx = \
                        utils.get_nearest_element_and_index(era5_lats,
                                                            pt_lats[i])
                row['x'] = lon_idx - self.half_reg_edge_grid_intervals
                row['y'] = lat_idx - self.half_reg_edge_grid_intervals
                diff_mins = pt_dts[i] 
                row['satel_era5_diff_mins'] = diff_mins[y][x]

        return env_data

    def get_env_data_of_matrix_with_coordinates(
        self, tc, col_names, era5_lons, era5_lats, file_path=None):
        """

        """
        env_data = []
        lats_num = len(era5_lats)
        lons_num = len(era5_lons)

        draw_region = [min(era5_lats), max(era5_lats),
                       min(era5_lons), max(era5_lons)]
        # Get `satel_era5_diff_mins`
        if 'smap' in self.sources:
            smap_lons, smap_lats, diff_mins = \
                    utils.get_xyz_matrix_of_smap_windspd_or_diff_mins(
                        'diff_mins', file_path, tc.date_time,
                        draw_region)
            if diff_mins is None:
                breakpoint()
                return None
            # Not all points of area has this feature
            # Make sure it is interpolated properly
            diff_mins = utils.interp_satel_era5_diff_mins_matrix(
                diff_mins)
        elif 'sfmr' in self.sources:
            diff_mins = np.zeros(shape=(lats_num, lons_num), dtype=float)

        for y in range(lats_num):
            for x in range(lons_num):
                row = dict()
                for name in col_names:
                    row[name] = None

                row['x'] = x - self.half_reg_edge_grid_intervals
                row['y'] = y - self.half_reg_edge_grid_intervals
                row['lon'] = era5_lons[x]
                row['lat'] = era5_lats[y]
                row['satel_era5_diff_mins'] = diff_mins[y][x]

                env_data.append(row)

        return env_data

    def get_smap_prediction_lonlat(self, tc):
        success, lat1, lat2, lon1, lon2 = \
                utils.get_subset_range_of_grib(
                    tc.lat, tc.lon, self.lats['era5'],
                    self.lons['era5'], self.reg_edge)
        if not success:
            return False, None, None

        era5_lons = [
            x * self.spa_resolu['era5']['atm'] + lon1 for x in range(
                int((lon2 - lon1) / self.spa_resolu['era5']['atm']) + 1)
        ]
        era5_lats = [
            y * self.spa_resolu['era5']['atm'] + lat1 for y in range(
                int((lat2 - lat1) / self.spa_resolu['era5']['atm']) + 1)
        ]

        return True, era5_lons, era5_lats

    def add_era5_single_levels_data(self, env_data, col_names, tc,
                                    area):
        rounded_dt = utils.hour_rounder(tc.date_time)
        hourtime = rounded_dt.hour

        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        try:
            era5_file_path = era5_manager.download_single_levels_vars(
                'tc', tc.date_time, '', [hourtime], area, 'smap',
                tc.sid, show_info=False)
        except Exception as msg:
            self.logger.error((
                f"""Fail downloading ERA5 single levels vars in """
                f"""function add_era5_single_levels_data: {msg}"""))
            return 'exit', None

        north, west, south, east = area
        grbs = pygrib.open(era5_file_path)
        messages_num = grbs.messages
        grbs.close()
        total = messages_num
        count = 0

        grbidx = pygrib.index(era5_file_path, 'dataTime')
        selected_grbs = grbidx.select(dataTime=hourtime*100)

        spa_resolu_diff = (self.spa_resolu['era5']['ocean']
                           - self.spa_resolu['era5']['atm'])

        for grb in selected_grbs:
            # Generate name which is the same with table column
            name = utils.process_grib_message_name(grb.name)
            grb_spa_resolu = grb.jDirectionIncrementInDegrees

            data, lats, lons = grb.data(south, north, west, east)
            # type of data may be numpy.ma.core.MaskedArray
            data = np.flip(data, 0)
            lats = np.flip(lats, 0)
            lons = np.flip(lons, 0)

            new_data = None
            if grb_spa_resolu == self.spa_resolu['era5']['ocean']:
                # Sharpen ocean grid data despite is masked or not
                new_data, new_lats, new_lons = \
                        utils.sharpen_era5_ocean_grid(data, lats, lons)
            # Now data is np.ndarray or np.ma.core.MaskedArray.
            # Traverse data to fill fileds in dict `env_data`.
            # If data at one point is masked, then remove the
            # corresponding row in dict `env_data`

            # MUST check masked array like this, because if an array
            # is numpy.ma.core.MaskedArray, it is numpy.ndarray too.
            # So only directly check whether an array is instance
            # of numpy.ma.core.MaskedArray is safe.
            if isinstance(data, np.ma.core.MaskedArray):
                if new_data is None:
                    # original data is atmosphere variable
                    env_data = self.add_masked_array_grb(
                        env_data, name, data, lats, lons)
                else:
                    # original data is oceanic variable
                    env_data = self.add_masked_array_grb(
                        env_data, name, new_data, new_lats,
                        new_lons)
            else:
                if new_data is None:
                    # original data is atmosphere variable
                    env_data = self.add_ndarray_grb(
                        env_data, name, data, lats, lons)
                else:
                    # original data is oceanic variable
                    env_data = self.add_ndarray_grb(
                        env_data, name, new_data, new_lats,
                        new_lons)

        pres_lvls = []
        for row in env_data:
            nearest_pres_lvl, nearest_pres_lvl_idx = \
                    utils.get_nearest_element_and_index(
                        self.pres_lvls,
                        row['mean_sea_level_pressure'] / 100)

            windspd, winddir = utils.compose_wind(
                row['neutral_wind_at_10_m_u_component'],
                row['neutral_wind_at_10_m_v_component'],
                'o')
            row['era5_10m_neutral_equivalent_windspd'] = windspd
            row['era5_10m_neutral_equivalent_winddir'] = winddir

            pres_lvls.append(nearest_pres_lvl)

        return env_data, pres_lvls

    def add_ndarray_grb(self, env_data, grb_name, data,
                        lats, lons):
        for row in env_data:
            try:
                lat = row['lat']
                lon = row['lon']
                lat_idx = np.where(lats==lat)[0][0]
                lon_idx = np.where(lons==lon)[1][0]
                value = data[lat_idx][lon_idx]

                row[grb_name] = value
            except Exception as msg:
                breakpoint()
                exit(msg)

        return env_data

    def add_masked_array_grb(self, env_data, grb_name, data, 
                             lats, lons):
        result = []

        for row in env_data:
            try:
                lat = row['lat']
                lon = row['lon']
                lat_idx = np.where(lats==lat)[0][0]
                lon_idx = np.where(lons==lon)[1][0]

                if not data.mask[lat_idx][lon_idx]:
                    value = data[lat_idx][lon_idx]
                    row[grb_name] = value
                    result.append(row)
            except Exception as msg:
                breakpoint()
                exit(msg)

        return result

    def add_era5_pressure_levels_data(self, env_data, col_names, tc,
                                      area, pres_lvls):
        rounded_dt = utils.hour_rounder(tc.date_time)
        hourtime = rounded_dt.hour

        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        try:
            era5_file_path = era5_manager.download_pressure_levels_vars(
                'tc', tc.date_time, '', [hourtime], area,
                sorted(list(set(pres_lvls))), 'smap', tc.sid,
                show_info=False)
        except Exception as msg:
            self.logger.error((
                f"""Fail downloading ERA5 pressure levels vars in """
                f"""function add_era5_pressure_levels_data: {msg}"""))
            return 'exit'

        north, west, south, east = area
        grbs = pygrib.open(era5_file_path)
        messages_num = grbs.messages
        grbs.close()
        total = messages_num
        count = 0

        grbidx = pygrib.index(era5_file_path, 'dataTime')
        selected_grbs = grbidx.select(dataTime=hourtime*100)

        spa_resolu_diff = (self.spa_resolu['era5']['ocean']
                           - self.spa_resolu['era5']['atm'])

        for grb in selected_grbs:
            # Generate name which is the same with table column
            name = utils.process_grib_message_name(grb.name)
            grb_spa_resolu = grb.jDirectionIncrementInDegrees

            data, lats, lons = grb.data(south, north, west, east)
            # type of data may be numpy.ma.core.MaskedArray
            data = np.flip(data, 0)
            lats = np.flip(lats, 0)
            lons = np.flip(lons, 0)

            # MUST check masked array like this, because if an array
            # is numpy.ma.core.MaskedArray, it is numpy.ndarray too.
            # So only directly check whether an array is instance
            # of numpy.ma.core.MaskedArray is safe.
            if isinstance(data, np.ma.core.MaskedArray):
                env_data = self.add_masked_array_grb(env_data, grb_name,
                                                     data, lats, lons)
            else:
                env_data = self.add_ndarray_grb(env_data, name, data,
                                                lats, lons)

        return env_data

    def get_smap_xyz_matrix(self, tc, era5_lons, era5_lats):
        satel_manager = satel_scs.SCSSatelManager(
            self.CONFIG, self.period, self.region, self.db_root_passwd,
            save_disk=self.save_disk, work=False)
        smap_file_path = satel_manager.download('smap', tc.date_time)
        if smap_file_path is None:
            return False, None, None, None, None

        draw_region = [min(era5_lats), max(era5_lats),
                       min(era5_lons), max(era5_lons)]
        lons, lats, windspd = \
                utils.get_xyz_matrix_of_smap_windspd_or_diff_mins(
                    'windspd', smap_file_path, tc.date_time, draw_region)

        smap_lons, smap_lats, diff_mins = \
                utils.get_xyz_matrix_of_smap_windspd_or_diff_mins(
                    'diff_mins', smap_file_path, tc.date_time,
                    draw_region)

        if (windspd is None
            or not utils.satel_data_cover_tc_center(
                lons, lats, windspd, tc)):
            return False, None, None, None, None, None
        else:
            return True, lons, lats, windspd, utils.if_mesh(lons), \
                    diff_mins

    def get_ccmp_xyz_matrix(self, tc):
        ccmp_manager = ccmp.CCMPManager(self.CONFIG, self.period,
                                        self.region, self.db_root_passwd,
                                        # work_mode='fetch')
                                        work_mode='')
        ccmp_file_path = ccmp_manager.download_ccmp_on_one_day(
            tc.date_time)

        lons, lats, windspd = utils.get_xyz_matrix_of_ccmp_windspd(
            ccmp_file_path, tc.date_time, self.region)

        if windspd is not None:
            return True, lons, lats, windspd, utils.if_mesh(lons)
        else:
            self.logger.info((f"""No CCMP data on {tc.date_time}"""))
            return False, None, None, None, None

    def get_era5_xyz_matrix(self, tc, era5_lons, era5_lats):
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        rounded_dt = utils.hour_rounder(tc.date_time)
        if rounded_dt.day != tc.date_time.day:
            return False, None, None, None, None, None

        # North, West, South, East,
        area = [max(era5_lats), min(era5_lons),
                min(era5_lats), max(era5_lons)]
        # South, North, West, East
        draw_region = [min(era5_lats), max(era5_lats),
                       min(era5_lons), max(era5_lons)]
        match_satel = None
        for src in self.sources:
            for satel_name in self.CONFIG['satel_names']:
                if src.startswith(satel_name):
                    match_satel = src
                    break
        if match_satel is None:
            match_satel = 'era5'
            # return False, None, None, None, None

        try:
            era5_file_path = era5_manager.download_single_levels_vars(
                'surface_wind', tc.date_time, '', [rounded_dt.hour],
                area, match_satel, tc.sid, show_info=False)
        except Exception as msg:
            self.logger.error((
                f"""Fail downloading ERA5 single levels vars in """
                f"""function get_era5_xyz_matrix: {msg}"""))
            return False, 'exit', None, None, None, None

        lons, lats, windspd = utils.get_xyz_matrix_of_era5_windspd(
            era5_file_path, 'single_levels', tc.date_time, draw_region)

        return True, lons, lats, windspd, utils.if_mesh(lons), None
