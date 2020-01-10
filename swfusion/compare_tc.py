import datetime
import logging
import os
import pickle
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
        self.compare_instructions = compare_instructions
        self.basin = basin

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.sources = ['ccmp', 'era5', 'ascat', 'wsat', 'amsr2',
                        'smap', 'sentinel_1']

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

        self.zorders = self.CONFIG['plot']['zorders']['scs_basemap']

        utils.reset_signal_handler()

        self.sources = ['smap_prediction', 'sfmr']
        self.compare_sources()

    def compare_sources(self):
        self.logger.info((f"""Comparing wind speed from different sources"""))
        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name'][
            self.basin]
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)

        sources_str = ''
        for idx, src in enumerate(self.sources):
            if idx < len(self.sources) - 1:
                sources_str = f"""{sources_str}{src.upper()} and """
            else:
                sources_str = f"""{sources_str}{src.upper()}"""

        tc_query = self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        )
        total = tc_query.count()
        # Filter TCs during period
        for idx, tc in enumerate(tc_query):
            if tc.wind < 64:
                continue
            # if tc.lat < 10 or tc.lat > 20:
            #     continue
            # if tc.lon < 110 or tc.lon > 120:
            #     continue
            if tc.r34_ne is None:
                continue
            converted_lon = utils.longtitude_converter(
                tc.lon, '360', '-180')
            if bool(globe.is_land(tc.lat, converted_lon)):
                continue
            # Draw windspd from CCMP, ERA5, Interium
            # and several satellites
            success = False
            if 'sfmr' not in self.sources:
                success = self.compare_with_one_tc_record(tc)
            else:
                if idx < total - 1:
                    next_tc = tc_query[idx + 1]
                    if tc.sid == next_tc.sid:
                        success = self.compare_with_sfmr(
                            tc, next_tc)
            if success:
                print((f"""Comparing {sources_str} with IBTrACS record """
                       f"""of TC {tc.name} on {tc.date_time}"""))
            else:
                print((f"""Skiping comparsion of {sources_str} with """
                       f"""IBTrACS record of TC {tc.name} """
                       f"""on {tc.date_time}"""))
        print('Done')

    def compare_with_sfmr(self, tc, next_tc):
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are far away in time
        if delta.days:
            return False
        # Check existence of SFMR between two IBTrACS records
        existence, spatial_temporal_info = self.sfmr_exists(tc, next_tc)
        if not existence:
            return False

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
        if len(hour_info_pt_idx):
            breakpoint()
        else:
            return False

        # For each hour, compare SFMR wind speed with regressed wind speed
        for h in hours:
            interp_tc = self.interp_tc(h, tc, next_tc)
            success = self.compare_with_sfmr_around_interped_tc(
                spatial_temporal_info, hour_info_pt_idx[h], interp_tc)
            if not success:
                return False

    def compare_with_sfmr_around_interped_tc(
        self, sfmr_brief_info, one_hour_info_pt_idx, interp_tc):
        # Reference function `compare_with_one_tc_record`
        subplots_row, subplots_col = utils.get_subplots_row_col(
            len(self.sources))
        fig, axs = plt.subplots(subplots_row, subplots_col,
                                figsize=(15, 7), sharey=True)
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

        smap_lons, smap_lats = self.get_smap_prediction_lonlat(interp_tc)
        # North, West, South, East
        area = [max(smap_lats), min(smap_lons),
                min(smap_lats), max(smap_lons)]

        lons = dict()
        lats = dict()
        windspd = dict()
        radii_area = dict()
        mesh = dict()

        max_windspd = -1
        for idx, src in enumerate(self.sources):
            ax = axs_list[idx]

            # Get lons, lats, windspd
            success, lons[src], lats[src], windspd[src], mesh[src] = \
                    self.get_sources_xyz(src, interp_tc, ax, smap_lons,
                                         smap_lats, sfmr_brief_info,
                                         one_hour_info_pt_idx)
            if not success:
                return False
            # Get max windspd
            if windspd[src].max() > max_windspd:
                max_windspd = windspd[src].max()

        # Draw windspd
        for idx, src in enumerate(self.sources):
            try:
                ax = axs_list[idx]

                utils.draw_windspd(self, fig, ax, tc.date_time,
                                   lons[src], lats[src], windspd[src],
                                   max_windspd, mesh[src], custom=True,
                                   area=area)
                ax.text(-0.1, 1.025, f'({string.ascii_lowercase[idx]})',
                        transform=ax.transAxes, fontsize=16,
                        fontweight='bold', va='top', ha='right')
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

        return True

    def sfmr_rounded_hours(self, tc, next_tc, spatial_temporal_info):
        # Include start hour, but not end hour
        # To let all intervals same
        hours_between_two_tcs = []
        datetime_area = dict()
        hour_info_pt_idx = dict()

        delta = next_tc.date_time - tc.date_time
        hours = int(delta.seconds / 3600)
        for h in range(hours):
            interp_dt = tc.date_time + datetime.timedelta(
                seconds = h * 3600)
            hours_between_two_tcs.append(interp_dt)

            datetime_area[interp_dt] = dict()
            interp_tc = self.interp_tc(h, tc, next_tc)
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
                f"""{year}/{info.hurr_name}/{info.file_name}"""
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
                pt_datetime = datetime.datetime.combine(
                    utils.sfmr_nc_converter('DATE', vars['DATE'][i]),
                    utils.sfmr_nc_converter('TIME', vars['TIME'][i])
                )
                rounded_hour = utils.hour_rounder(pt_datetime)

                # Check whether rounded hours are in hours between two TCs
                if rounded_hour not in hours_between_two_tcs:
                    continue


                lon = (vars['LON'][i] + 360) % 360
                lat = vars['LAT'][i]

                # Check whether SFMR data points are in area around TC at rounded
                # hour
                if (lon < datetime_area[rounded_hour]['lon1']
                    or lon > datetime_area[rounded_hour]['lon2']
                    or lat < datetime_area[rounded_hour]['lat1']
                    or lat > datetime_area[rounded_hour]['lat2']):
                    continue
                rounded_hour_idx = hours_between_two_tcs.index(rounded_hour)

                # Add SFMR data point index into `hour_info_pt_idx`
                if rounded_hour not in hour_info_pt_idx:
                    hour_info_pt_idx[rounded_hour] = dict()
                if info_idx not in hour_info_pt_idx[rounded_hour]:
                    hour_info_pt_idx[rounded_hour][info_idx] = []

                hour_info_pt_idx[rounded_hour][info_idx].append(i)

        return hour_info_pt_idx

    def sfmr_exists(self, tc, next_tc):
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
        BriefInfo = utils.get_class_by_tablename(self.engine, table_name)

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
        lon_shift, lat_shift = utils.get_center_shift_of_two_tcs(
            next_tc, tc)
        hourly_lon_shift = lon_shift / hours
        hourly_lat_shift = lat_shift / hours
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
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        hours = int(delta.seconds / 3600)
        # Spatial shift
        lon_shift, lat_shift = utils.get_center_shift_of_two_tcs(
            next_tc, tc)
        hourly_lon_shift = lon_shift / hours
        hourly_lat_shift = lat_shift / hours

        interp_tc = tc
        # Only interpolate `date_time`, `lon`, `lat` variables
        # Other variables stays same with `tc`
        interp_tc.date_time = tc.date_time + datetime.timedelta(
            seconds = h * 3600)
        interp_tc.lon = (h * hourly_lon_shift + tc.lon)
        interp_tc.lat = (h * hourly_lat_shift + tc.lat)

        return interp_tc

    def compare_with_one_tc_record(self, tc):
        subplots_row, subplots_col = utils.get_subplots_row_col(
            len(self.sources))
        fig, axs = plt.subplots(subplots_row, subplots_col,
                                figsize=(15, 7), sharey=True)
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

        smap_lons, smap_lats = self.get_smap_prediction_lonlat(tc)
        # North, West, South, East
        area = [max(smap_lats), min(smap_lons),
                min(smap_lats), max(smap_lons)]

        lons = dict()
        lats = dict()
        windspd = dict()
        radii_area = dict()
        mesh = dict()

        max_windspd = -1
        for idx, src in enumerate(self.sources):
            ax = axs_list[idx]

            # Get lons, lats, windspd
            success, lons[src], lats[src], windspd[src], mesh[src] = \
                    self.get_sources_xyz(src, tc, ax, smap_lons,
                                         smap_lats)
            if not success:
                return False
            # Get max windspd
            if windspd[src].max() > max_windspd:
                max_windspd = windspd[src].max()
            # Draw wind radii
            radii_area[src] = utils.draw_ibtracs_radii(ax, tc,
                                                       self.zorders)

        # Draw windspd
        for idx, src in enumerate(self.sources):
            try:
                ax = axs_list[idx]

                utils.draw_windspd(self, fig, ax, tc.date_time,
                                   lons[src], lats[src], windspd[src],
                                   max_windspd, mesh[src], custom=True,
                                   area=area)
                ax.text(-0.1, 1.025, f'({string.ascii_lowercase[idx]})',
                        transform=ax.transAxes, fontsize=16,
                        fontweight='bold', va='top', ha='right')
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

        return True

    def get_sources_xyz(self, src, tc, ax, smap_lons=None,
                        smap_lats=None, sfmr_brief_info=None,
                        one_hour_info_pt_idx=None):
        if src == 'ccmp':
            return self.get_ccmp_xyz(tc, ax)
        elif src == 'era5':
            return self.get_era5_xyz(tc, ax)
        elif src == 'smap':
            return self.get_smap_xyz(tc, ax)
        elif src == 'smap_prediction':
            return self.get_smap_prediction_xyz(tc, ax, smap_lons,
                                                smap_lats)
        elif src == 'sfmr':
            return self.get_sfmr_xyz(tc, ax, smap_lons, smap_lats,
                                     sfmr_brief_info,
                                     one_hour_info_pt_idx)

    def get_sfmr_xyz(self, tc, ax, grid_lons, grid_lats,
                     sfmr_brief_info, one_hour_info_pt_idx):
        success = False
        sfmr_lons = grid_lons
        sfmr_lats = grid_lats
        sfmr_windspd = np.full(shape=(len(grid_lats), len(grid_lons)),
                               fill_value=-1, dtype=float)

        # Logger information
        self.logger.info(f'Getting xyz of SFMR around TC')

        # Get SFMR windspd
        for info_idx in one_hour_info_pt_idx.keys():
            brief_info = sfmr_brief_info[info_idx]
            data_indices = one_hour_info_pt_idx[info_idx]
            # Firstly try first-come-first-count method
            # Secondly try square-average method
            sfmr_windspd = utils.update_sfmr_windspd_matrix(
                grid_lons, grid_lats, sfmr_windspd, brief_info.file_path,
                data_indices)

        return (True, sfmr_lons, sfmr_lats, sfmr_windspd,
                utils.if_mesh(sfmr_lons))

    def get_smap_prediction_xyz(self, tc, ax, smap_lons, smap_lats):
        self.logger.info(f'Getting xyz of SMAP prediction around TC')
        # Test if era5 data can be extracted
        rounded_dt = utils.hour_rounder(tc.date_time)
        if rounded_dt.day != tc.date_time.day:
            return False, None, None, None, None

        # Create a new dataframe to store all points in region
        # Each point consists of ERA5 vars and SMAP windspd to predict

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

        # smap_lons, smap_lats = self.get_smap_prediction_lonlat(tc)
        # North, West, South, East,
        area = [max(smap_lats), min(smap_lons),
                min(smap_lats), max(smap_lons)]
        env_data = self.get_env_data_with_coordinates(
            col_names, smap_lats, smap_lons)

        # Get single levels vars of ERA5 around TC
        env_data, pres_lvls = self.add_era5_single_levels_data(
            env_data, col_names, tc, area)
        if env_data is None:
            return False, None, None, None, None

        # Get pressure levels vars of ERA5 around TC
        env_data = self.add_era5_pressure_levels_data(
            env_data, col_names, tc, area, pres_lvls)

        # Predict SMAP windspd
        env_df = pd.DataFrame(env_data, columns=col_names,
                              dtype=np.float64)
        # Get original lon and lat before normalization
        # DO NOT use to_numpy() method, because it is a pointer to
        # DataFrame column, which changes with pointed column
        smap_windspd_xyz_dict = {
            'lon': list(env_df['lon']),
            'lat': list(env_df['lat']),
        }
        # Normalize data if the best model is trained after normalization
        if 'no-normalization' not in self.compare_instructions:
            scaler = MinMaxScaler() 
            env_df[env_df.columns] = scaler.fit_transform(
                env_df[env_df.columns])

        model_dir = self.CONFIG['regression']['dirs']['tc']\
                ['xgboost']['model']
        best_model = utils.load_best_xgb_model(model_dir)
        smap_windspd_xyz_dict['windspd'] = list(
            best_model.predict(data=env_df))
        smap_windspd_xyz_df = pd.DataFrame(smap_windspd_xyz_dict,
                                           dtype=np.float64)
        # Pad SMAP windspd prediction around TC to all region
        smap_windspd = self.padding_tc_windspd_prediction(
            smap_windspd_xyz_df, smap_lons, smap_lats)

        # Return data
        return (True, smap_lons, smap_lats, smap_windspd,
                utils.if_mesh(smap_lons))

    def padding_tc_windspd_prediction(self, smap_windspd_xyz_df,
                                      smap_lons, smap_lats):
        df = smap_windspd_xyz_df
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

        return windspd

    def get_env_data_with_coordinates(self, col_names, smap_lats,
                                      smap_lons):
        env_data = []
        lats_num = len(smap_lats)
        lons_num = len(smap_lons)

        for y in range(lats_num):
            for x in range(lons_num):
                row = dict()
                for name in col_names:
                    row[name] = None

                row['x'] = x - self.half_reg_edge_grid_intervals
                row['y'] = y - self.half_reg_edge_grid_intervals
                row['lon'] = smap_lons[x]
                row['lat'] = smap_lats[y]

                env_data.append(row)

        return env_data

    def get_smap_prediction_lonlat(self, tc):
        success, lat1, lat2, lon1, lon2 = \
                utils.get_subset_range_of_grib(
                    tc.lat, tc.lon, self.lats['era5'],
                    self.lons['era5'], self.reg_edge)

        smap_lons = [
            x * self.spa_resolu['era5']['atm'] + lon1 for x in range(
                int((lon2 - lon1) / self.spa_resolu['era5']['atm']) + 1)
        ]
        smap_lats = [
            y * self.spa_resolu['era5']['atm'] + lat1 for y in range(
                int((lat2 - lat1) / self.spa_resolu['era5']['atm']) + 1)
        ]

        return smap_lons, smap_lats

    def add_era5_single_levels_data(self, env_data, col_names, tc, area):
        rounded_dt = utils.hour_rounder(tc.date_time)
        if rounded_dt.day != tc.date_time.day:
            return None
        hourtime = rounded_dt.hour

        self.logger.info((f"""Adding data from ERA5 single levels """
                          f"""reanalysis"""))
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = \
                era5_manager.download_single_levels_vars(
                    'tc', tc.date_time, '', [hourtime], area, 'smap',
                    tc.sid)

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
        if rounded_dt.day != tc.date_time.day:
            return None
        hourtime = rounded_dt.hour

        self.logger.info((f"""Adding data from ERA5 pressure levels """
                          f"""reanalysis"""))
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = \
                era5_manager.download_pressure_levels_vars(
                    'tc', tc.date_time, '', [hourtime], area,
                    sorted(list(set(pres_lvls))), 'smap', tc.sid)

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

    def get_smap_xyz(self, tc, ax):
        satel_manager = satel_scs.SCSSatelManager(
            self.CONFIG, self.period, self.region, self.db_root_passwd,
            save_disk=self.save_disk, work=False)
        smap_file_path = satel_manager.download('smap', tc.date_time)

        lons, lats, windspd = utils.get_xyz_of_smap_windspd(
            smap_file_path, tc.date_time, self.region)

        if (windspd is None
            or not utils.satel_data_cover_tc_center(
                lons, lats, windspd, tc)):
            return False, None, None, None, None
        else:
            return True, lons, lats, windspd, utils.if_mesh(lons)

    def get_ccmp_xyz(self, tc, ax):
        ccmp_manager = ccmp.CCMPManager(self.CONFIG, self.period,
                                        self.region, self.db_root_passwd,
                                        # work_mode='fetch')
                                        work_mode='')
        ccmp_file_path = ccmp_manager.download_ccmp_on_one_day(
            tc.date_time)

        lons, lats, windspd = utils.get_xyz_of_ccmp_windspd(
            ccmp_file_path, tc.date_time, self.region)

        if windspd is not None:
            return True, lons, lats, windspd, utils.if_mesh(lons)
        else:
            self.logger.info((f"""No CCMP data on {tc.date_time}"""))
            return False, None, None, None, None

    def get_era5_xyz(self, tc, ax):
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        rounded_dt = utils.hour_rounder(tc.date_time)
        if rounded_dt.day != tc.date_time.day:
            return False, None, None, None, None

        # smap_lons, smap_lats = self.get_smap_prediction_lonlat(tc)
        # area = [max(smap_lats), min(smap_lons),
        #         min(smap_lats), max(smap_lons)]

        # North, West, South, East,
        area = [self.lat2, self.lon1, self.lat1, self.lon2]
        match_satel = None
        for src in self.sources:
            for satel_name in self.CONFIG['satel_names']:
                if src.startswith(src):
                    match_satel = src
                    break
        if match_satel is None:
            return False, None, None, None, None

        era5_file_path = era5_manager.download_single_levels_vars(
            'surface_wind', tc.date_time, '', [rounded_dt.hour], area,
            match_satel, tc.sid)

        lons, lats, windspd = utils.get_xyz_of_era5_windspd(
            era5_file_path, 'single_levels', tc.date_time, self.region)

        return True, lons, lats, windspd, utils.if_mesh(lons)
