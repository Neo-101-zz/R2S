import calendar
import copy
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
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import or_
import netCDF4

import ccmp
import era5
import satel_scs
import utils
import match_era5_smap

Base = declarative_base()

class TCComparer(object):

    def __init__(self, CONFIG, period, region, basin, passwd,
                 save_disk, compare_instructions, draw_sfmr, work=True):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.save_disk = save_disk
        self.engine = None
        self.session = None
        self.compare_instructions = list(set(compare_instructions))
        self.basin = basin
        self.draw_sfmr = draw_sfmr

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.spa_resolu = dict()

        self.spa_resolu['ccmp'] = self.CONFIG['ccmp'][
            'spatial_resolution']
        self.spa_resolu['era5'] = dict()
        self.spa_resolu['era5']['atm'] = self.CONFIG['era5'][
            'spatial_resolution']
        self.spa_resolu['era5']['ocean'] = self.CONFIG['era5'][
            'ocean_spatial_resolution']

        self.lats = dict()
        self.lons = dict()

        self.lats['era5'] = [y * self.spa_resolu['era5']['atm'] - 90
                             for y in range(721)]
        self.lons['era5'] = [x * self.spa_resolu['era5']['atm']
                             for x in range(1440)]

        self.spa_resolu['smap'] = self.CONFIG['rss'][
            'spatial_resolution']
        self.lats['smap'] = [
            y * self.spa_resolu['smap'] - 89.875 for y in range(720)]
        self.lons['smap'] = [
            x * self.spa_resolu['smap'] + 0.125 for x in range(1440)]

        self.source_candidates = ['smap_prediction', 'smap', 'era5',
                                  'sfmr', 'ibtracs', 'merra2']
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

        if 'sfmr' in self.sources and 'ibtracs' in self.sources:
            self.logger.error((f"""Do not support comparison """
                               f"""simultaneously involving """
                               f"""SFMR and IBTrACS"""))
            exit()

        # Swap 'sfmr' and first source string when necessary
        if 'sfmr' in self.sources and self.sources.index('sfmr') != 0:
            tmp = self.sources[0]
            des_idx = self.sources.index('sfmr')
            self.sources[0] = 'sfmr'
            self.sources[des_idx] = tmp

        if set(['sfmr', 'smap', 'smap_prediction', 'era5']).issubset(
                set(self.sources)):
            self.sources = ['sfmr', 'era5', 'smap', 'smap_prediction']

        # `smap` should before `smap_prediction` to filter cases
        # where they did not synchronously exists because the cases
        # where SMAP overlapped SFMR around TC is less frequent than
        # that  of SMAP prediction
        # if set(['sfmr', 'smap', 'smap_prediction']).issubset(
        #         set(self.sources)):
        #     smap_index = self.sources.index('smap')
        #     smap_pred_index = self.sources.index('smap_prediction')
        #     if smap_index > smap_pred_index:
        #         self.sources[smap_index], self.sources[smap_pred_index]\
        #             = self.sources[smap_pred_index], self.sources[
        #                 smap_index]

        if 'sfmr' in self.sources and len(self.sources) < 2:
            self.logger.error((
                f"""At least must input 1 sources """
                f"""when not including SFMR"""))
            exit()

        self.sources_titles = self.CONFIG['plot'][
            'data_sources_title']

        self.sources_str = self.gen_sources_str()

        self.reg_edge = self.CONFIG['regression']['edge_in_degree']

        self.half_reg_edge = self.reg_edge / 2
        self.half_reg_edge_grid_intervals = int(
            self.half_reg_edge / self.spa_resolu['era5']['atm'])

        self.pres_lvls = self.CONFIG['era5']['pres_lvls']

        # self.grid_lons = None
        # self.grid_lats = None
        # self.grid_x = None
        # self.grid_y = None
        # Load 4 variables above
        if not hasattr(self, 'grid_lons') or self.grid_lons is None:
            utils.load_grid_lonlat_xy(self)

        self.zorders = self.CONFIG['plot']['zorders']['compare']

        utils.reset_signal_handler()

        if work:
            self.compare_sources()

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
        tc_query = self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        )
        total = tc_query.count()

        # Filter TCs during period
        for idx, tc in enumerate(tc_query):
            converted_lon = utils.longitude_converter(tc.lon,
                                                      '360', '-180')
            if bool(globe.is_land(tc.lat, converted_lon)):
                continue
            # Draw windspd from different sources
            success = False
            if 'sfmr' in self.sources:
                if idx < total - 1:
                    next_tc = tc_query[idx + 1]
                    if tc.sid == next_tc.sid:
                        success = self.compare_with_sfmr(tc, next_tc)

            elif 'ibtracs' in self.sources:
                if (tc.wind is not None and tc.pres is not None
                        and not tc.date_time.minute
                        and not tc.date_time.second):
                    success = self.compare_with_ibtracs(tc)
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
                success, need_exit = \
                    self.compare_betweem_two_tc_records(tc, next_tc)

        print('Done')

    def compare_with_ibtracs(self, tc):
        # Get max windspd from sources except IBTrACS
        success, smap_lons, smap_lats = \
                self.get_smap_lonlat(tc)
        if not success:
            return False
        del success

        lons = dict()
        lats = dict()
        windspd = dict()
        radii_area = dict()
        mesh = dict()
        diff_mins = dict()

        for src in self.sources:
            max_windspd = -1
            if src == 'ibtracs':
                continue
            try:
                # Get lons, lats, windspd
                success, lons[src], lats[src], windspd[src], \
                        mesh[src], diff_mins[src] = \
                        self.get_sources_xyz_matrix(
                            src, tc, smap_lons, smap_lats)
                if not success:
                    return False
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

            print((f"""Validate {src.upper()} with IBTrACS of TC """
                   f"""{tc.name} on {tc.date_time}"""))
            Validation = utils.create_ibtracs_validation_table(
                self, src)
            row = Validation()
            row.tc_sid = tc.sid
            row.ibtracs_datetime = tc.date_time
            row.ibtracs_windspd_mps = 0.514444 * tc.wind
            row.ibtracs_pres_mb = tc.pres

            if src in ['era5', 'smap_prediction']:
                src_dt = tc.date_time
            elif src == 'smap':
                avg_diff_mins = float(diff_mins[src].mean())
                src_dt = tc.date_time + datetime.timedelta(
                    seconds=int(60*avg_diff_mins))
            else:
                self.logger.error('Do not know datetime of source')
                exit()

            setattr(row, f'{src}_datetime', src_dt)
            setattr(row, f'{src}_windspd_mps', max_windspd)

            row.tc_sid_ibtracs_datetime = (
                f'{row.tc_sid}_{row.ibtracs_datetime}')

            utils.bulk_insert_avoid_duplicate_unique(
                [row], self.CONFIG['database']\
                ['batch_size']['insert'],
                Validation, ['tc_sid_ibtracs_datetime'],
                self.session, check_self=True)

        return True

    def compare_with_sfmr(self, tc, next_tc):
        # Skip interpolating between two TC records if the begin and
        # termianl of interval is not at the hour
        if (next_tc.date_time.minute or next_tc.date_time.second
                or tc.date_time.minute or tc.date_time.second):
            return False
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC records if two neighbouring
        # records of TC are far away in time
        if delta.days:
            return False
        hours = int(delta.seconds / 3600)
        # Time interval between two TCs are less than one hour
        if not hours:
            return False

        if len(self.sources) > 2:
            # Exploit match tables of SFMR and another source which
            # have been created before to quickly check whether
            # the match of SFMR and at least two other sources exists
            for src in self.sources[1:]:
                MatchOneSrc = utils.create_match_table(
                    self, ['sfmr', src])
                result = utils.match_exists_during_tc_interval(
                    self, hours, tc, next_tc, MatchOneSrc)
                if not result['success']:
                    return False

        Match = utils.create_match_table(self, self.sources)
        result = utils.match_exists_during_tc_interval(
            self, hours, tc, next_tc, Match)
        if not result['success']:
            return False

        if len(result['hit_dt']) == hours:
            # In this condition, `match count` is larger than zero.  
            # Just compare the datetimes when `match` value is True
            # and no need to update `match_data_sources`
            success = self.compare_with_sfmr_all_records_hit(
                tc, next_tc, hours, result['match_dt'],
                result['spatial_temporal_info'],
                result['hour_info_pt_idx'])
        else:
            # First executed here between particular two TCs.  
            # For each hour, compare SFMR wind speed with wind speed
            # from other sources and have to update `match_data_sources`
            success = self.compare_with_sfmr_not_all_records_hit(
                tc, next_tc, hours, result['spatial_temporal_info'], 
                result['hour_info_pt_idx'])

        return success

    def compare_with_sfmr_not_all_records_hit(self, tc, next_tc, hours,
                                              spatial_temporal_info,
                                              hour_info_pt_idx):
        Match = utils.create_match_table(self, self.sources)

        for h in range(hours):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)

            if interped_tc.date_time not in hour_info_pt_idx.keys():
                # update corrseponding match
                utils.update_one_row_of_match(self, Match, interped_tc,
                                              False)
                print((f"""[Not exist] SFMR of TC {tc.name} near """
                       f"""{interped_tc.date_time}"""))
                continue

            success = self.compare_with_sfmr_around_interped_tc(
                spatial_temporal_info,
                hour_info_pt_idx[interped_tc.date_time],
                interped_tc)

            if success:
                utils.update_one_row_of_match(self, Match,
                                              interped_tc, True)
                print((f"""[Match] {self.sources_str} """
                       f"""of TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))
            else:
                utils.update_one_row_of_match(self, Match,
                                              interped_tc, False)
                print((f"""[Not match] {self.sources_str} """
                       f"""of TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))

        return True

    def compare_with_sfmr_all_records_hit(
        self, tc, next_tc, hours, match_dt, spatial_temporal_info,
        hour_info_pt_idx):
        #
        for h in range(hours):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)

            if interped_tc.date_time not in match_dt:
                print((f"""[Skip] {self.sources_str} """
                       f"""of TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))
                continue

            success = self.compare_with_sfmr_around_interped_tc(
                spatial_temporal_info,
                hour_info_pt_idx[interped_tc.date_time],
                interped_tc)
            print((f"""[Redo] {self.sources_str} """
                   f"""of TC {interped_tc.name} """
                   f"""on {interped_tc.date_time}"""))
            if not success:
                self.logger.error('Match True but not success')
                continue

        return True

    def compare_with_sfmr_around_interped_tc(
        self, sfmr_brief_info, one_hour_info_pt_idx, interped_tc):
        # Reference function `compare_betweem_two_tc_records`
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

        success, smap_lons, smap_lats = \
            self.get_smap_lonlat(interped_tc)
        if not success:
            return False
        del success
        # North, South, West, East
        draw_region = [min(smap_lats), max(smap_lats),
                       min(smap_lons), max(smap_lons)]

        lons = dict()
        lats = dict()
        windspd = dict()
        mesh = dict()
        diff_mins = dict()

        max_windspd = -1
        min_windspd = 999
        for src in self.sources:
            if src == 'sfmr':
                success, sfmr_tracks, sfmr_pts = \
                        utils.average_sfmr_along_track(
                            self, interped_tc, sfmr_brief_info,
                            one_hour_info_pt_idx)
                if not success:
                    return False
                for single_track_pts in sfmr_pts:
                    for pt in single_track_pts:
                        max_windspd = max(pt.windspd, max_windspd)
                        min_windspd = min(pt.windspd, min_windspd)
            else:
                try:
                    # Get lons, lats, windspd
                    success, lons[src], lats[src], windspd[src], \
                            mesh[src], diff_mins[src] = \
                            self.get_sources_xyz_matrix(
                                src, interped_tc, smap_lons,
                                smap_lats)
                    if not success:
                        return False
                    if ((src == 'smap_prediction'
                         and 'smap' in self.sources)
                        or src == 'smap'):
                        # Not all points of area has this feature
                        # Make sure it is interpolated properly
                        diff_mins[src] = utils.\
                                interp_satel_era5_diff_mins_matrix(
                                    diff_mins[src])
                    # Get max windspd
                    if windspd[src].max() > max_windspd:
                        max_windspd = windspd[src].max()
                    if windspd[src].min() > min_windspd:
                        min_windspd = windspd[src].min()
                except Exception as msg:
                    breakpoint()
                    exit(msg)

        if 'smap' in self.sources and 'smap_prediction' in self.sources:
            tag = 'aligned_with_smap'
            # Compare 2D wind field of SMAP and SMAP prediction
            utils.compare_2d_sources(self, interped_tc, lons, lats,
                                     windspd, mesh, diff_mins, 'smap',
                                     'smap_prediction')
        else:
            tag = None
        # To quantificationally validate simulated SMAP wind,
        # we need to extract SMAP points matching SFMR points
        for src in self.sources:
            if src == 'sfmr':
                continue
            sfmr_pts = utils.validate_with_sfmr(
                self, src, interped_tc, sfmr_pts,
                lons[src], lats[src], windspd[src], mesh[src],
                diff_mins[src], tag)

        sfmr_pts_null = True
        for t in range(len(sfmr_pts)):
            if len(sfmr_pts[t]):
                sfmr_pts_null = False
                break
        if sfmr_pts_null:
            return False

        index = 0
        tc_dt = interped_tc.date_time
        if (('smap_prediction' in self.sources
             and 'smap' in self.sources)
                or 'smap' in self.sources):
            accurate_dt = tc_dt + datetime.timedelta(
                seconds=60*diff_mins['smap'].mean())
        else:
            accurate_dt = tc_dt
        subplot_title_suffix = {
            'smap': f'{accurate_dt.strftime("%H%M UTC %d %b %Y")} ',
            'era5': f'{tc_dt.strftime("%H%M UTC %d %b %Y")} ',
        }
        subplot_title_suffix['smap_prediction'] = subplot_title_suffix[
            'smap']

        # Draw windspd
        for src in self.sources:
            try:
                if src == 'sfmr':
                    continue
                ax = axs_list[index]

                fontsize = 20
                utils.draw_windspd(self, fig, ax, interped_tc.date_time,
                                   lons[src], lats[src], windspd[src],
                                   max_windspd, mesh[src],
                                   draw_contour=False,
                                   custom=True, region=draw_region)
                if self.draw_sfmr:
                    utils.draw_sfmr_windspd_and_track(
                        self, fig, ax, interped_tc.date_time,
                        sfmr_tracks, sfmr_pts, max_windspd)
                if text_subplots_serial_number:
                    ax.text(0.1, 0.95,
                            f'{string.ascii_lowercase[index]})',
                            transform=ax.transAxes, fontsize=20,
                            fontweight='bold', va='top', ha='right')
                ax.set_title((f"""{self.sources_titles[src]} """
                              f"""{subplot_title_suffix[src]} """
                              f"""{interped_tc.name} """),
                             size=15)
                index += 1
            except Exception as msg:
                breakpoint()
                exit(msg)

        fig.tight_layout(pad=0.1)

        dt_str = interped_tc.date_time.strftime('%Y_%m%d_%H%M')
        fig_dir = self.CONFIG['result']['dirs']['fig']['root']
        for idx, src in enumerate(self.sources):
            if not idx:
                fig_dir = f"""{fig_dir}{src}_vs"""
            elif idx < len(self.sources) - 1:
                fig_dir = f"""{fig_dir}_{src}_vs"""
            else:
                fig_dir = f"""{fig_dir}_{src}/"""

        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'{dt_str}_{interped_tc.name}.eps'
        plt.savefig(f'{fig_dir}{fig_name}', dpi=600)
        plt.clf()

        return True

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
            interped_tc = utils.interp_tc(self, h, tc, next_tc)
            tc_aft = tc
            if tc.date_time == next_tc.date_time:
            # if tc_pre.date_time != tc_aft.date_time:
                breakpoint()

            datetime_area[interp_dt]['lon1'] = \
                    interped_tc.lon - self.half_reg_edge
            datetime_area[interp_dt]['lon2'] = \
                    interped_tc.lon + self.half_reg_edge
            datetime_area[interp_dt]['lat1'] = \
                    interped_tc.lat - self.half_reg_edge
            datetime_area[interp_dt]['lat2'] = \
                    interped_tc.lat + self.half_reg_edge

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
        half_reg_edge = self.reg_edge / 2
        corners = {'left': [], 'top': [], 'right': [], 'bottom': []}
        # Extract from the interval between two TC records
        for h in range(hours):
            interped_tc_lon = (h * hourly_lon_shift + tc.lon)
            interped_tc_lat = (h * hourly_lat_shift + tc.lat)
            corners['left'].append(interped_tc_lon - half_reg_edge)
            corners['top'].append(interped_tc_lat + half_reg_edge)
            corners['right'].append(interped_tc_lon + half_reg_edge)
            corners['bottom'].append(interped_tc_lat - half_reg_edge)
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
            # ATTENTIONL: DO NOT direct use `interped_tc = tc`
            # Because it makes a link between two variables
            # any modification will simultaneously change two variables
            interped_tc = IBTrACS()
            interped_tc.sid = tc.sid
            interped_tc.name = tc.name
            # interped_tc.basin = tc.basin
            # interped_tc.pres = tc.pres
            # interped_tc.wind = tc.wind
            # interped_tc.r34_ne = tc.r34_ne
            # interped_tc.r34_se = tc.r34_se
            # interped_tc.r34_sw = tc.r34_sw
            # interped_tc.r34_nw = tc.r34_nw
            # interped_tc.r50_ne = tc.r50_ne
            # interped_tc.r50_ne = tc.r50_ne
            # interped_tc.r50_se = tc.r50_se
            # interped_tc.r50_sw = tc.r50_sw
            # interped_tc.r64_nw = tc.r64_nw
            # interped_tc.r64_se = tc.r64_se
            # interped_tc.r64_sw = tc.r64_sw
            # interped_tc.r64_nw = tc.r64_nw
            # Only interpolate `date_time`, `lon`, `lat` variables
            # Other variables stays same with `tc`
            interped_tc.date_time = tc.date_time + datetime.timedelta(
                seconds = h * 3600)
            interped_tc.lon = (h * hourly_lon_shift + tc.lon)
            interped_tc.lat = (h * hourly_lat_shift + tc.lat)
        except Exception as msg:
            breakpoint()
            exit(msg)

        return interped_tc

    def compare_betweem_two_tc_records(self, tc, next_tc):
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are far away in time
        if delta.days:
            return False
        hours = int(delta.seconds / 3600)
        # Time interval between two TCs are less than one hour
        if not hours:
            return False

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

        success, smap_lons, smap_lats = \
            self.get_smap_lonlat(tc)
        if not success:
            return False, False
        del success
        # South, North, West, East
        draw_region = [min(smap_lats), max(smap_lats),
                       min(smap_lons), max(smap_lons)]

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
                    src, tc, smap_lons, smap_lats)
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

    def get_sources_xyz_matrix(self, src, tc, smap_lons=None,
                               smap_lats=None):
        if src == 'ccmp':
            return self.get_ccmp_xyz_matrix(tc)
        elif src == 'era5':
            return self.get_era5_xyz_matrix(tc, smap_lons, smap_lats)
        elif src == 'smap':
            return self.get_smap_xyz_matrix(tc, smap_lons, smap_lats)
        elif src == 'smap_prediction':
            return self.get_smap_prediction_xyz_matrix(
                tc, smap_lons, smap_lats)

    def get_smap_prediction_xyz_matrix(self, tc, smap_lons, smap_lats):
        return self.get_smap_prediction_xyz_matrix_step_1(
            tc, smap_lons, smap_lats)

    def get_smap_prediction_xyz_matrix_step_1(self, tc, smap_lons,
                                              smap_lats):
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
            useless_cols_name = self.CONFIG['regression'][
                'useless_columns']['smap_era5']
            for col in cols:
                if col.name not in useless_cols_name:
                    col_names.append(col.name)

            # smap_lons, smap_lats = self.get_smap_lonlat(tc)
            # North, West, South, East,
            area = [max(smap_lats), min(smap_lons),
                    min(smap_lats), max(smap_lons)]

            return self.get_smap_prediction_xyz_matrix_step_2(
                tc, smap_lons, smap_lats, col_names, area)
        except Exception as msg:
            self.logger.error((
                f"""function get_smap_prediction_xyz_matrix_step_1:"""
                f""" {msg}"""))
            breakpoint()
            return False, 'exit', None, None, None, None

    def get_smap_prediction_xyz_matrix_step_2(self, tc, smap_lons,
                                              smap_lats, col_names,
                                              area):
        try:
            suffix = f'_{tc.sid}_{tc.date_time.strftime("%Y%m%d%H%M%S")}'
            SMAPERA5 = utils.create_smap_era5_table(self, tc.date_time,
                                                    suffix)
            # Need set the temporal shift from era5 as same as original
            # SMAP
            if 'smap' in self.sources:
                satel_manager = satel_scs.SCSSatelManager(
                    self.CONFIG, self.period, self.region,
                    self.db_root_passwd, save_disk=self.save_disk,
                    work=False)
                smap_file_path = satel_manager.download('smap',
                                                        tc.date_time)

                env_data, diff_mins = \
                    self.get_env_data_of_matrix_with_coordinates(
                        tc, col_names, smap_lons, smap_lats, SMAPERA5,
                        smap_file_path)

                match_manager = match_era5_smap.matchManager(
                    self.CONFIG, self.period, self.region, self.basin,
                    self.db_root_passwd, False, work=False)

                smap_data, hourtimes, smap_area = match_manager.\
                    get_smap_part(SMAPERA5, tc, smap_file_path)
            # Just set the temporal shift from era5 to zero
            else:
                env_data, diff_mins = \
                    self.get_env_data_of_matrix_with_coordinates(
                        tc, col_names, smap_lons, smap_lats, SMAPERA5)
                hourtimes = [utils.hour_rounder(tc.date_time).hour]

            if env_data is None:
                return False, None, None, None, None, None

            diff = 0.5
            north = max(smap_lats)
            south = min(smap_lats)
            east = max(smap_lons)
            west = min(smap_lons)
            area = [north + diff, (west - diff + 360) % 360,
                    south - diff, (east + diff + 360) % 360]

            for idx, val in enumerate(area):
                if utils.is_multiple_of(val, 0.125):
                    # decrease north and east a little
                    if not idx or idx == 3:
                        area[idx] = val - 0.125
                    # increase west and south a little
                    else:
                        area[idx] = val + 0.125
                else:
                    self.logger.error(('Coordinates of SMAP are not '
                                        f'multiple of 0.125'))
            area[1] = (area[1] + 360) % 360
            area[3] = (area[3] + 360) % 360

            env_data = utils.add_era5(self, 'smap', tc, env_data,
                                      hourtimes, area)

            utils.bulk_insert_avoid_duplicate_unique(
                env_data, self.CONFIG['database']['batch_size'][
                    'insert'],
                SMAPERA5, ['satel_datetime_lon_lat'], self.session,
                check_self=True)

            table_name = utils.gen_tc_satel_era5_tablename('smap',
                                                           self.basin)
            table_name += suffix

            df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)

            # Compare generated ERA5 data with that matched with SMAP wind
            """
            refer_table_name = f'tc_smap_era5_{self.basin}'
            ReferTable = utils.get_class_by_tablename(
                self.engine, refer_table_name)
            refer_query = self.session.query(ReferTable).filter(
                ReferTable.era5_datetime == tc.date_time)
            refer_num = refer_query.count()

            not_compare_cols = copy.copy(self.CONFIG['regression'][
                'useless_columns']['smap_era5'])
            not_compare_cols.append('smap_windspd')
            compare_cols = list(set(df.columns.tolist())
                                - set(not_compare_cols))
            if refer_num:
                diff_count = {
                    'pressure': {},
                    'single': {},
                }
                diff_percent_sum = {
                    'pressure': {},
                    'single': {},
                }
                for col in self.CONFIG['era5']['vars']['grib'][
                        'reanalysis_pressure_levels']['all_vars']:
                    if col in compare_cols:
                        diff_count['pressure'][col] = 0
                        diff_percent_sum['pressure'][col] = 0
                for col in self.CONFIG['era5']['vars']['grib'][
                        'reanalysis_single_levels']['tc']:
                    if col in compare_cols:
                        diff_count['single'][col] = 0
                        diff_percent_sum['single'][col] = 0

                # Compare `env_data` and `refer_query`
                for i in range(refer_num):
                    refer = refer_query[i]
                    same_pt_env = df.loc[(df['x'] == refer.x)
                                         & (df['y'] == refer.y)]
                    if not len(same_pt_env):
                        continue

                    for col in compare_cols:
                        env_val = same_pt_env[col].iloc[0]
                        refer_val = getattr(refer, col)
                        diff = env_val - refer_val
                        avg = (env_val - refer_val) / 2.0

                        if diff:
                            # TODO: check whether pressure levels vars
                            # biased
                            if col in self.CONFIG['era5']['vars'][
                                'grib']['reanalysis_pressure_levels'][
                                    'all_vars']:
                                diff_count['pressure'][col] += 1
                                diff_percent_sum['pressure'][col] += (
                                    diff / avg)
                            elif col in self.CONFIG['era5']['vars'][
                                'grib']['reanalysis_single_levels'][
                                    'tc']:
                                diff_count['single'][col] += 1
                                diff_percent_sum['single'][col] += (
                                    diff / avg)
                utils.show_diff_count(self, diff_count,
                                      diff_percent_sum)
            """
            # drop table
            utils.drop_table_by_name(self.engine, self.session,
                                     table_name)

            # Get original lon and lat.  It is possible that lon and lat
            # are not in `col_names`.  So we should extract them
            # specifically.
            env_lons = []
            env_lats = []
            # XXX
            for i in range(len(df)):
                env_lons.append(df['lon'][i])
                env_lats.append(df['lat'][i])
            # for i in range(len(env_data)):
            #     env_lons.append(env_data[i].lon)
            #     env_lats.append(env_data[i].lat)

            smap_windspd_xyz_matrix_dict = {
                'lon': env_lons,
                'lat': env_lats,
            }

            df.drop(self.CONFIG['regression']['useless_columns'][
                'smap_era5'], axis=1, inplace=True)

            env_df = df.drop(['smap_windspd'],axis=1).reset_index(
                drop=True)

            model_dirs = {
                'SMOGN-TCL': (
                    '/Users/lujingze/Programming/SWFusion/'
                    'regression/tc/lightgbm/model/'
                    'na_valid_2557.909583_fl_smogn_final_thre_'
                    '50_power_3_under_maxeval_100/'),
                'MSE': ('/Users/lujingze/Programming/SWFusion/'
                        'regression/tc/lightgbm/model/'
                        'na_valid_2.496193/'),
                'FL-CLF': ('/Users/lujingze/Programming/SWFusion/'
                           'classify/tc/lightgbm/model/'
                           'na_valid_0.560000_45_fl_smogn_final'
                           '_unb_maxeval_2/'),
            }
            models = dict()
            preds = dict()
            for idx, (key, val) in enumerate(model_dirs.items()):
                save_file = [f for f in os.listdir(val)
                             if f.endswith('.pkl')
                             and f.startswith(f'{self.basin}')]
                if len(save_file) != 1:
                    self.logger.error('Count of Bunch is not ONE')
                    exit(1)

                with open(f'{val}{save_file[0]}', 'rb') as f:
                    models[key] = (pickle.load(f)).model

                preds[key] = models[key].predict(env_df)

            y_pred = np.zeros(shape=(env_df.shape[0],))
            for i in range(env_df.shape[0]):
                if preds['FL-CLF'][i] > 0:
                    y_pred[i] = preds['SMOGN-TCL'][i]
                else:
                    y_pred[i] = preds['MSE'][i]

            # model_dir = ('/Users/lujingze/Programming/SWFusion/'
            #              'regression/tc/lightgbm/model/'
            #              'na_0.891876_fl_smogn_50_slope_0.05/')
            # # best_model = utils.load_best_model(model_dir, self.basin)
            # model = utils.load_model_from_bunch(model_dir, self.basin,
            #                                     '.pkl', 'model')
            # y_pred = model.predict(env_df)
            smap_windspd_xyz_matrix_dict['windspd'] = list(y_pred)

            smap_windspd_xyz_matrix_df = pd.DataFrame(
                smap_windspd_xyz_matrix_dict, dtype=np.float64)
            # Pad SMAP windspd prediction around TC to all region
            smap_windspd = self.padding_tc_windspd_prediction(
                smap_windspd_xyz_matrix_df, smap_lons, smap_lats)
        except Exception as msg:
            breakpoint()
            exit(msg)

        # Return data
        return (True, smap_lons, smap_lats, smap_windspd,
                utils.if_mesh(smap_lons), diff_mins)

    def padding_tc_windspd_prediction(self,
                                      smap_windspd_xyz_matrix_df,
                                      smap_lons, smap_lats):
        df = smap_windspd_xyz_matrix_df
        windspd = np.full(shape=(len(smap_lats), len(smap_lons)),
                          fill_value=-1, dtype=float)

        for index, row in df.iterrows():
            try:
                lat = row['lat']
                lon = row['lon']
                lat_idx = smap_lats.index(lat)
                lon_idx = smap_lons.index(lon)
                windspd[lat_idx][lon_idx] = row['windspd']
            except Exception as msg:
                breakpoint()
                exit(msg)

        windspd = ma.masked_values(windspd, -1)
        return windspd

    def get_env_data_of_matrix_with_coordinates(self, tc,
                                                col_names,
                                                smap_lons,
                                                smap_lats,
                                                SMAPERA5,
                                                file_path=None):
        """

        """
        env_data = []
        lats_num = len(smap_lats)
        lons_num = len(smap_lons)

        draw_region = [min(smap_lats), max(smap_lats),
                       min(smap_lons), max(smap_lons)]
        # Need set the temporal shift from era5 as same as original
        # SMAP
        if 'smap' in self.sources:
            smap_lons, smap_lats, diff_mins = \
                    utils.get_xyz_matrix_of_smap_windspd_or_diff_mins(
                        'diff_mins', file_path, tc.date_time,
                        draw_region)
            if diff_mins is None:
                return None, None
            # Not all points of area has this feature
            # Make sure it is interpolated properly
            diff_mins = utils.interp_satel_era5_diff_mins_matrix(
                diff_mins)
        else:
            if tc.date_time.minute == 0 and tc.date_time.second == 0:
                # Just set the temporal shift from era5 to zero
                diff_mins = np.zeros(shape=(lats_num, lons_num),
                                     dtype=float)
            else:
                # When tc is not at the hour
                diff_mins_val = int((tc.date_time - utils.hour_rounder(
                    tc.date_time)).total_seconds() / 60)
                diff_mins = np.full(shape=(lats_num, lons_num),
                                    fill_value=diff_mins_val,
                                    dtype=int)

        for y in range(lats_num):
            for x in range(lons_num):
                row = SMAPERA5()

                row.sid = tc.sid
                row.satel_era5_diff_mins = diff_mins[y][x]
                row.era5_datetime = utils.hour_rounder(tc.date_time)
                row.satel_datetime = (
                    row.era5_datetime + datetime.timedelta(
                        seconds=int(row.satel_era5_diff_mins)*60))
                row.x = x - self.half_reg_edge_grid_intervals
                row.y = y - self.half_reg_edge_grid_intervals
                # Simulate SMAP lon and lat
                row.lon = smap_lons[x]
                row.lat = smap_lats[y]
                row.satel_datetime_lon_lat = (
                    f"""{row.satel_datetime}"""
                    f"""_{row.lon}_{row.lat}""")
                row.smap_windspd = 0

                env_data.append(row)

        return env_data, diff_mins

    def get_smap_lonlat(self, tc):
        success, lat1, lat2, lon1, lon2 = \
            utils.get_subset_range_of_grib(
                tc.lat, tc.lon, self.lats['smap'],
                self.lons['smap'], self.reg_edge)
        if not success:
            return False, None, None

        smap_lons = [
            x * self.spa_resolu['smap'] + lon1 for x in range(
                int((lon2-lon1) / self.spa_resolu['smap']) + 1)
        ]
        smap_lats = [
            y * self.spa_resolu['smap'] + lat1 for y in range(
                int((lat2-lat1) / self.spa_resolu['smap']) + 1)
        ]

        return True, smap_lons, smap_lats

    def get_smap_xyz_matrix(self, tc, smap_lons, smap_lats):
        satel_manager = satel_scs.SCSSatelManager(
            self.CONFIG, self.period, self.region,
            self.db_root_passwd,
            save_disk=self.save_disk, work=False)
        smap_file_path = satel_manager.download('smap', tc.date_time)
        if smap_file_path is None:
            return False, None, None, None, None, None

        draw_region = [min(smap_lats), max(smap_lats),
                       min(smap_lons), max(smap_lons)]
        lons, lats, windspd = \
            utils.get_xyz_matrix_of_smap_windspd_or_diff_mins(
                'windspd', smap_file_path, tc.date_time,
                draw_region)

        lons, lats, diff_mins = \
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

    def get_era5_xyz_matrix(self, tc, smap_lons, smap_lats):
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
        area = [max(smap_lats), min(smap_lons),
                min(smap_lats), max(smap_lons)]
        # South, North, West, East
        draw_region = [min(smap_lats), max(smap_lats),
                       min(smap_lons), max(smap_lons)]
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
            exit()

        lons, lats, windspd = utils.get_xyz_matrix_of_era5_windspd(
            era5_file_path, 'single_levels', tc.date_time,
            draw_region)

        return True, lons, lats, windspd, utils.if_mesh(lons), None
