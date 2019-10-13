"""Manage downloading and reading ASCAT, QucikSCAT and Windsat data.

"""
import datetime
import math
import logging
import pickle
import os
import time
import sys

from geopy import distance
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import mapper
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from matplotlib.patches import Polygon
import netCDF4
import pygrib

import utils
import cwind
import sfmr
import era5

MASKED = np.ma.core.masked
Base = declarative_base()

class SatelManager(object):
    """Manage features of satellite data that are not related to other data
    sources except TC table from IBTrACS.

    """

    def __init__(self, CONFIG, period, region, passwd,
                 spatial_window, temporal_window):
        self.logger = logging.getLogger(__name__)
        self.satel_names = ['smap', 'amsr2', 'ascat', 'qscat', 'wsat']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]

        self.edge = self.CONFIG['rss']['subset_edge_in_degree']
        self.spa_resolu = self.CONFIG['rss']['spatial_resolution']
        self.lat_grid_points = [
            y * self.spa_resolu - 89.875 for y in range(
                self.CONFIG['rss']['lat_grid_points_number'])
        ]
        self.lon_grid_points = [
            x * self.spa_resolu + 0.125 for x in range(
                self.CONFIG['rss']['lon_grid_points_number'])
        ]
        self.era5_lat_grid_points = [
            y * self.spa_resolu - 90 for y in range(
                self.CONFIG['era5']['lat_grid_points_number'])
        ]
        self.era5_lon_grid_points = [
            x * self.spa_resolu for x in range(
                self.CONFIG['era5']['lon_grid_points_number'])
        ]

        # Size of 3D grid points around TC center
        self.grid_2d = dict()
        self.grid_2d['lat_axis'] = self.grid_2d['lon_axis'] = int(
            self.edge/self.spa_resolu)
        self.missing_value = self.CONFIG['rss']['missing_value']

        utils.setup_database(self, Base)
        # self.download_and_read_tc_oriented()
        self.download_and_read_satel_era5()

    def _get_basic_columns_tc_oriented(self):
        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('date_time', DateTime, nullable=False))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('x_y', String(20), nullable=False,
                           unique=True))

        return cols

    def _get_satel_table_class_tc_oriented(self, satel_name, sid, dt):
        dt_str = dt.strftime('%Y_%m%d_%H%M')
        table_name = f'{satel_name}_tc_{sid}_{dt_str}'

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return table_name, None, Satel

        cols = self._get_basic_columns_tc_oriented()
        if satel_name == 'ascat' or satel_name == 'qscat':
            cols.append(Column('windspd', Float, nullable=False))
            cols.append(Column('winddir', Float, nullable=False))
            cols.append(Column('scatflag', Boolean, nullable=False))
            cols.append(Column('radrain', Float, nullable=False))
        elif satel_name == 'wsat':
            cols.append(Column('w_lf', Float, nullable=False))
            cols.append(Column('w_mf', Float, nullable=False))
            cols.append(Column('w_aw', Float, nullable=False))
            cols.append(Column('winddir', Float, nullable=False))
            cols.append(Column('sst', Float, nullable=False))
            cols.append(Column('vapor', Float, nullable=False))
            cols.append(Column('cloud', Float, nullable=False))
            cols.append(Column('rain', Float, nullable=False))
        elif satel_name == 'amsr2':
            cols.append(Column('windLF', Float, nullable=False))
            cols.append(Column('windMF', Float, nullable=False))
            cols.append(Column('sst', Float, nullable=False))
            cols.append(Column('vapor', Float, nullable=False))
            cols.append(Column('cloud', Float, nullable=False))
            cols.append(Column('rain', Float, nullable=False))
        elif satel_name == 'smap':
            cols.append(Column('windspd', Float, nullable=False))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        mapper(Satel, t)

        return table_name, t, Satel

    def _get_basic_columns_self_oriented(self):
        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('satel_datetime', DateTime, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('satel_datetime_lon_lat', String(70),
                           nullable=False, unique=True))

        return cols

    def _create_satel_era5_table_class(self, satel_name, dt):
        table_name = f'{satel_name}_{dt.year}'

        class SatelERA5(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(SatelERA5, t)

            return SatelERA5

        satel_cols = self._get_basic_columns_self_oriented()
        if satel_name == 'smap':
            satel_cols.append(Column('windspd', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, work=False)
        era5_cols = era5_.get_era5_columns()
        cols = satel_cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(SatelERA5, t)

        self.session.commit()

        return SatelERA5

    def download_and_read_satel_era5(self):
        utils.setup_signal_handler()

        dt_delta = self.period[1] - self.period[0]

        for day in range(dt_delta.days):
            target_datetime = (self.period[0]
                               + datetime.timedelta(days=day))
            self.logger.info((f'Download and reading satellite '
                              + f'and ERA5 data: '
                              + f'{target_datetime.date()}'))
            for satel_name in ['smap']:
                # Download satellite data
                satel_data_path = self.download(satel_name,
                                                target_datetime)
                utils.reset_signal_handler()
                if satel_data_path is None:
                    continue
                # Download ERA5 data
                self.logger.debug((f'Downloading all surface ERA5 '
                                   + f'data of all hours on '
                                   + f'{target_datetime.date()}'))
                era5_ = era5.ERA5Manager(self.CONFIG, self.period,
                                         self.region,
                                         self.db_root_passwd,
                                         work=False)
                era5_data_path = \
                        era5_.download_all_surface_vars_of_whole_day(
                            target_datetime)

                # Get satellite table
                SatelERA5 = self._create_satel_era5_table_class(
                    satel_name, target_datetime)

                self.logger.debug((f'Reading {satel_name} and ERA5 '
                                   + f'data on '
                                   + f'{target_datetime.date()}'))
                self.read_satel_era5(satel_name, satel_data_path,
                                     era5_data_path, SatelERA5)
                breakpoint()
                os.remove(era5_data_path)

    def read_satel_era5(self, satel_name, satel_data_path,
                        era5_data_path, SatelERA5):
        # Extract satel and ERA5 data
        if satel_name == 'smap':
            satel_era5_data = self.extract_smap_era5(
                satel_data_path, era5_data_path, SatelERA5)
        # Insert into table
        utils.bulk_insert_avoid_duplicate_unique(
            satel_era5_data, self.CONFIG['database']\
            ['batch_size']['insert'],
            SatelERA5, ['satel_datetime_lon_lat'], self.session,
            check_self=True)

    def extract_smap_era5(self, satel_data_path, era5_data_path,
                          SatelERA5):
        # Get SMAP part first
        self.logger.debug(f'Getting SMAP part from {satel_data_path}')
        smap_part = self._get_smap_part(satel_data_path, SatelERA5)

        # Then add ERA5 part
        self.logger.debug(f'Adding ERA5 part from {era5_data_path}')
        smap_era5_data = self._add_era5_part(smap_part, era5_data_path,
                                             test=2)
        return smap_era5_data

    def _get_smap_part(self, satel_data_path, SatelERA5):
        smap_part = []
        dataset = netCDF4.Dataset(satel_data_path)
        vars = dataset.variables

        date_str = satel_data_path.split(
            self.CONFIG['smap']['data_prefix'])[1][:10]\
                .replace('_', '')

        for y in range(self.CONFIG['rss']['lat_grid_points_number']):

            if not np.ma.MaskedArray.count(vars['minute'][y]):
                continue

            time_lon_index, time_iasc_index = np.ma.nonzero(
                vars['minute'][y])
            wind_lon_index, wind_iasc_index = np.ma.nonzero(
                vars['wind'][y])
            valid_lon_index, time_index, wind_index = np.intersect1d(
                time_lon_index, wind_lon_index, return_indices=True)

            for idx in wind_index:
                x = wind_lon_index[idx]
                i = wind_iasc_index[idx]

                row = SatelERA5()

                # Process the datetime and skip if necessary
                time_str ='{:02d}{:02d}00'.format(
                    *divmod(int(vars['minute'][y][x][i]), 60))

                if time_str.startswith('24'):
                    date_ = datetime.datetime.strptime(
                        date_str + '000000', '%Y%m%d%H%M%S').date()
                    time_ = datetime.time(0, 0, 0)
                    date_ = date_ + datetime.timedelta(days=1)
                    row.satel_datetime = datetime.datetime.combine(
                        date_, time_)
                else:
                    row.satel_datetime = datetime.datetime.strptime(
                        date_str + time_str, '%Y%m%d%H%M%S')

                row.lon = float(vars['lon'][x])
                row.lat = float(vars['lat'][y])
                row.satel_datetime_lon_lat = (f'{row.satel_datetime}_'
                                              + f'{row.lon}_{row.lat}')
                row.windspd = float(vars['wind'][y][x][i])

                smap_part.append(row)

        return smap_part

    def _get_hourtime_row_dict(self, smap_part):
        hourtime_row = dict()

        for hourtime in range(0, 2400, 100):
            hourtime_row[hourtime] = []

        for idx, row in enumerate(smap_part):
            closest_time = (100 * utils.hour_rounder(
                row.satel_datetime).hour)
            hourtime_row[closest_time].append(idx)

        return hourtime_row

    def _add_era5_part(self, smap_part, era5_data_path, test):
        """
        Attention
        ---------
        Break this function into several sub-functions is not
        recommended. Because it needs to deliver large arguments
        such as smap_part and grib message.

        """
        satel_era5_data = []
        grbidx = pygrib.index(era5_data_path, 'dataTime')
        count = 0

        info = f'Adding ERA5 from {era5_data_path}: '
        total = len(smap_part) * 16
        # Get temporal relation of grbs and rows first
        hourtime_row = self._get_hourtime_row_dict(smap_part)

        # For every hour, update corresponding rows with grbs
        for hourtime in range(0, 2400, 100):
            if not len(hourtime_row[hourtime]):
                continue
            selected_grbs = grbidx.select(dataTime=hourtime)

            for grb in selected_grbs:
                # data() method of pygrib is time-consuming
                # So apply it to global area then update all
                # smap part with grb of specific hourtime,
                # which using data() method as less as possible
                data, lats, lons = grb.data(-90, 90, 0, 360)
                data = np.flip(data, 0)

                # Generate name which is the same with table column
                name = grb.name.replace(" ", "_").lower()
                if name == 'vorticity_(relative)':
                    name = 'vorticity_relative'

                # Update all rows which matching this hourtime
                for idx in hourtime_row[hourtime]:
                    count += 1
                    print(f'\r{info} {count}/{total}', end='')

                    row = smap_part[idx]

                    satel_minute = (row.satel_datetime.hour * 60
                                     + row.satel_datetime.minute)
                    grb_minute = int(hourtime/100) * 60
                    row.mins_diff = grb_minute - satel_minute

                    lat1, lat2, lon1, lon2 = self._get_corners_of_cell(
                        row.lon, row.lat)
                    corners = []
                    for lat in [lat1, lat2]:
                        for lon in [lon1, lon2]:
                            lat_idx = self.era5_lat_grid_points.index(
                                lat)
                            lon_idx = self.era5_lon_grid_points.index(
                                lon)
                            corners.append(data[lat_idx][lon_idx])

                    value = float( sum(corners) / len(corners) )
                    setattr(row, name, value)

        utils.delete_last_lines()
        print('Done')
        return smap_part

    def _get_corners_of_cell(self, lon, lat):
        delta = self.CONFIG['rss']['spatial_resolution'] * 0.5
        lat1 = lat - delta
        lat2 = lat + delta
        lon1 = lon - delta
        lon2 = lon + delta

        return lat1, lat2, lon1, lon2

    def download_and_read_tc_oriented(self):
        """Download satellite data according to TC records from IBTrACS,
        then read it into satellite table.

        """
        utils.setup_signal_handler()

        # Get TC table and count its row number
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        tc_query = self.session.query(TCTable)
        total = tc_query.count()
        del tc_query
        count = 0
        info = (f'Downloading and reading satellite data according to '
                + f'TC records')
        self.logger.info(info)

        # Loop all TC records
        for row in self.session.query(TCTable).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            count += 1
            print(f'\r{info} {count}/{total}', end='')

            # Get TC datetime
            tc_datetime = row.date_time
            if not utils.check_period(tc_datetime, self.period):
                continue

            # Get hit result and range of satellite data matrix near
            # TC center
            hit, lat1, lat2, lon1, lon2 = \
                    utils.get_subset_range_of_grib(
                        row.lat, row.lon, self.lat_grid_points,
                        self.lon_grid_points, self.edge)
            if not hit:
                continue

            for satel_name in self.satel_names:

                if satel_name == 'smap':
                    data_type = 'netcdf'
                else:
                    data_type = 'bytemap'

                # Get satellite table
                table_name, sa_table, SatelTable = \
                        self._get_satel_table_class_tc_oriented(
                            satel_name, row.sid, tc_datetime)
                # Download satellite data according to TC datetime
                data_path = self.download(satel_name, tc_datetime)
                utils.reset_signal_handler()

                if data_path is None:
                    continue

                # Show the TC area and sateliite data area in the
                # temporal window
                hit_count = self.show_match(lat1, lat2, lon1, lon2,
                                            satel_name, row, data_path,
                                            data_type, draw=True,
                                            temporal_restrict=True)
                if not hit_count[self.edge]:
                    continue

                # Read satellite data according to TC datetime
                self.read_tc_oriented(satel_name, data_path, data_type,
                                      tc_datetime, table_name,
                                      sa_table, SatelTable,
                                      lat1, lat2, lon1, lon2)

        utils.delete_last_lines()
        print('Done')

    def draw_tc_area(self, map, lat1, lat2, lon1, lon2, zorder):

        llcrnrlon = lon1
        urcrnrlon = lon2
        llcrnrlat = lat1
        urcrnrlat = lat2
        lower_left = (llcrnrlon, llcrnrlat)
        lower_right = (urcrnrlon, llcrnrlat)
        upper_left = (llcrnrlon, urcrnrlat)
        upper_right = (urcrnrlon, urcrnrlat)

        self.plot_rec(map, lower_left, upper_left, lower_right, upper_right, 10)

    def get_satel_coverage(self, map, tc_row, satel_name, data_path,
                           data_type, draw, temporal_restrict):
        if data_type == 'bytemap':
            count = self.get_bytemap_coverage(map, tc_row, satel_name,
                                              data_path, draw,
                                              temporal_restrict)
        elif data_type == 'netcdf':
            count = self.get_netcdf_coverage(map, tc_row, satel_name,
                                             data_path, draw,
                                             temporal_restrict)

        return count

    def get_netcdf_coverage(self, map, tc_row, satel_name, data_path,
                            draw, temporal_restrict):

        if satel_name == 'smap':
            count = self.get_smap_coverage(map, tc_row, satel_name,
                                           data_path, draw,
                                           temporal_restrict)

        return count

    def get_smap_coverage(self, map, tc_row, satel_name, data_path,
                          draw, temporal_restrict):
        dataset = netCDF4.Dataset(data_path)
        vars = dataset.variables
        lats = []
        lons = []

        date_str = data_path.split(
            self.CONFIG['smap']['data_prefix'])[1][:10]\
                .replace('_', '')

        for y in range(self.CONFIG['rss']['lat_grid_points_number']):

            if not np.ma.MaskedArray.count(vars['minute'][y]):
                continue

            time_lon_index, time_iasc_index = np.ma.nonzero(
                vars['minute'][y])
            wind_lon_index, wind_iasc_index = np.ma.nonzero(
                vars['wind'][y])
            valid_lon_index, time_index, wind_index = np.intersect1d(
                time_lon_index, wind_lon_index, return_indices=True)

            for idx in wind_index:
                x = wind_lon_index[idx]
                i = wind_iasc_index[idx]

                if (temporal_restrict
                    and not self.check_smap_match(
                        vars, y, x, i, tc_row.date_time, date_str)
                   ):
                    continue

                lats.append(vars['lat'][y])
                lons.append(vars['lon'][x])

                if draw:
                    self.draw_coverage_within_window(map, vars,
                                                     y, x)

        count = self.get_coverage_in_ranges(satel_name, tc_row,
                                            lats, lons)

        return count

    def check_smap_match(self, vars, y, x, i, tc_datetime,
                         date_str):
        # Process the datetime and skip if necessary
        time_str ='{:02d}{:02d}00'.format(
            *divmod(int(vars['minute'][y][x][i]), 60))

        try:
            if time_str.startswith('24'):
                date_ = datetime.datetime.strptime(
                    date_str + '000000', '%Y%m%d%H%M%S').date()
                time_ = datetime.time(0, 0, 0)
                date_ = date_ + datetime.timedelta(days=1)
                datetime_ = datetime.datetime.combine(
                    date_, time_)
            else:
                datetime_ = datetime.datetime.strptime(
                    date_str + time_str, '%Y%m%d%H%M%S')
        except Exception as msg:
            breakpoint()
            exit(msg)

        # Calculate the time delta between satellite datetime
        # and TC datetime
        if tc_datetime > datetime_:
            delta = tc_datetime - datetime_
        else:
            delta = datetime_ - tc_datetime
        # Skip this turn of loop if timedelta is largers than
        # presetted temporal window
        if delta.seconds > \
           self.CONFIG['match']['temporal_window']:
            return False

        return True

    def get_bytemap_coverage(self, map, tc_row, satel_name, data_path,
                             draw, temporal_restrict):
        dataset = utils.dataset_of_daily_satel(satel_name, data_path)
        vars = dataset.variables

        data_name = data_path.split('/')[-1]
        date_str = data_name.split('_')[1][:8]

        lats = []
        lons = []

        for i in range(self.CONFIG['rss']['passes_number']):
            for j in range(self.CONFIG['rss']['lat_grid_points_number']):
                for k in range(self.CONFIG['rss']['lon_grid_points_number']):

                    if (temporal_restrict
                        and not self.check_bytemap_match(
                            vars, i, j, k, tc_row.date_time,
                            date_str)
                       ):
                        continue
                    lats.append(vars['latitude'][j])
                    lons.append(vars['longitude'][k])

                    if draw:
                        self.draw_coverage_within_window(map, vars, j, k)

        count = self.get_coverage_in_ranges(satel_name, tc_row, lats, lons)

        return count

    def get_coverage_in_ranges(self, satel_name, tc_row, lats, lons):
        edges = [20, 16, 12, 8, 4]
        count = dict()
        for idx, e in enumerate(edges):
            count[e] = 0
            hit, lat1, lat2, lon1, lon2 = utils.get_subset_range_of_grib(
                tc_row.lat, tc_row.lon, self.lat_grid_points,
                self.lon_grid_points, e)
            if not hit:
                continue

            for i in range(len(lats)):
                if (lats[i] > lat1 and lats[i] < lat2
                    and lons[i] > lon1 and lons[i] < lon2):
                    count[e] += 1

            if not count[e]:
                for idx in range(idx+1, len(edges)):
                    count[edges[idx]] = 0
                break

            info = (f'{satel_name} {tc_row.sid} '
                    + f'{tc_row.name} '
                    + f'{tc_row.date_time} {e}: {count[e]}')
            self.logger.info(info)

        Coverage = self.create_satel_coverage_count_table(satel_name,
                                                          tc_row)
        row = Coverage()
        row.sid = tc_row.sid
        row.date_time = tc_row.date_time
        for e in edges:
            setattr(row, f'hit_{e}', count[e])
        row.sid_date_time = f'{tc_row.sid}_{tc_row.date_time}'

        utils.bulk_insert_avoid_duplicate_unique(
            [row], self.CONFIG['database']\
            ['batch_size']['insert'],
            Coverage, ['sid_date_time'], self.session,
            check_self=True)

        return count

    def create_satel_coverage_count_table(self, satel_name, tc_row):
        dt_str = tc_row.date_time.strftime('%Y_%m%d_%H%M')
        table_name = f'coverage_{satel_name}'
        edges = [20, 16, 12, 8, 4]

        class Coverage(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Coverage, t)

            return Coverage

        cols = []
        # IBTrACS columns
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('sid', String(13), nullable=False))
        cols.append(Column('date_time', DateTime, nullable=False))

        for e in edges:
            cols.append(Column(f'hit_{e}', Integer, nullable=False))

        cols.append(Column('sid_date_time', String(50), nullable=False,
                           unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(Coverage, t)

        self.session.commit()

        return Coverage

    def draw_coverage_within_window(self, map, vars, j, k):
        if 'latitude' in vars.keys() and 'lat' not in vars.keys():
            lat = vars['latitude'][j]
            lon = vars['longitude'][k] - 180
        elif 'lat' in vars.keys() and 'latitude' not in vars.keys():
            lat = vars['lat'][j]
            lon = vars['lon'][k] - 180

        grid_half_edge = (0.5 * self.CONFIG['rss']\
                          ['spatial_resolution'])

        lat1, lat2 = lat - grid_half_edge, lat + grid_half_edge
        lon1, lon2 = lon - grid_half_edge, lon + grid_half_edge
        lats = np.array([lat1, lat2, lat2, lat1])
        lons = np.array([lon1, lon1, lon2, lon2])
        self.draw_cloud_pixel(lats, lons, map, 'white')

    def check_bytemap_match(self, vars, i, j, k, tc_datetime,
                            date_str):
        if vars['nodata'][i][j][k]:
            return False

        if 'mingmt' in vars.keys():
            # Process the datetime and skip if necessary
            time_str ='{:02d}{:02d}00'.format(
                *divmod(int(vars['mingmt'][i][j][k]), 60))
        elif 'time' in vars.keys():
            # Process the datetime and skip if necessary
            time_str ='{:02d}{:02d}00'.format(
                *divmod(int(60*vars['time'][i][j][k]), 60))

        try:
            if time_str.startswith('24'):
                date_ = datetime.datetime.strptime(
                    date_str + '000000', '%Y%m%d%H%M%S').date()
                time_ = datetime.time(0, 0, 0)
                date_ = date_ + datetime.timedelta(days=1)
                datetime_ = datetime.datetime.combine(
                    date_, time_)
            else:
                datetime_ = datetime.datetime.strptime(
                    date_str + time_str, '%Y%m%d%H%M%S')
        except Exception as msg:
            breakpoint()
            exit(msg)

        # Calculate the time delta between satellite datetime
        # and TC datetime
        if tc_datetime > datetime_:
            delta = tc_datetime - datetime_
        else:
            delta = datetime_ - tc_datetime
        # Skip this turn of loop if timedelta is largers than
        # presetted temporal window
        if delta.seconds > \
           self.CONFIG['match']['temporal_window']:
            return False

        return True

    def show_match(self, lat1, lat2, lon1, lon2, satel_name,
                   tc_row, data_path, data_type, draw,
                   temporal_restrict):

        map = Basemap(llcrnrlon=-180.0, llcrnrlat=-90.0, urcrnrlon=180.0,
                      urcrnrlat=90.0)
        map.drawcoastlines()
        map.drawmapboundary(fill_color='aqua')
        map.fillcontinents(color='coral', lake_color='aqua')
        map.drawmeridians(np.arange(0, 360, 30))
        map.drawparallels(np.arange(-90, 90, 30))

        self.draw_tc_area(map, lat1, lat2, lon1, lon2, 10)


        count = self.get_satel_coverage(map, tc_row, satel_name,
                                        data_path, data_type, draw,
                                        temporal_restrict)
        if not draw:
            plt.clf()
            return count

        fig_path = (f'{self.CONFIG["result"]["dirs"]["fig"]}'
                     + f'data_match_on_{tc_row.date_time}_{satel_name}.png')
        self.logger.debug(f'Drawing {fig_path}')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path)
        plt.clf()

        return count

    def draw_cloud_pixel(self, lats, lons, mapplot, color):
        """Draw a pixel on the map. The fill color alpha level depends on the cloud index, 
        ranging from 0.1 (almost fully transparent) for confidently clear pixels to 1 (fully opaque)
        for confidently cloudy pixels.

        Keyword arguments:
        lats -- Array of latitude values for the pixel 4 corner points (numpy array)
        lons -- Array of longitudes values for the pixel 4 corner points (numpy array)
        index -- Cloud mask index for given pixel: 
            0: confidently_cloudy
            1: probably_cloudy
            2: probably_clear
            3: confidently_clear
        mapplot -- Map object for coordinate transformation

        Returns:
        None
        """
        # x, y = mapplot(lons, lats)
        # xy = zip(x,y)
        # poly = Polygon(xy, facecolor='white')
        pixel = []
        for i in range(len(lats)):
            pixel.append([lons[i], lats[i]])

        poly = Polygon(pixel, facecolor=color)
        plt.gca().add_patch(poly)

    def plot_rec(self, bmap, lower_left, upper_left, lower_right,
                 upper_right, zorder):
        xs = [lower_left[0], upper_left[0],
              lower_right[0], upper_right[0],
              lower_left[0], lower_right[0],
              upper_left[0], upper_right[0]]
        ys = [lower_left[1], upper_left[1],
              lower_right[1], upper_right[1],
              lower_left[1], lower_right[1],
              upper_left[1], upper_right[1]]
        bmap.plot(xs, ys, color='red', latlon = True, zorder=zorder)

    def read_tc_oriented(self, satel_name, data_path, data_type,
                         tc_datetime, table_name, sa_table, SatelTable,
                         lat1, lat2, lon1, lon2):

        self.logger.debug(f'Reading {satel_name}: {data_path}')

        lat1_idx = self.lat_grid_points.index(lat1)
        lat2_idx = self.lat_grid_points.index(lat2)
        lon1_idx = self.lon_grid_points.index(lon1)
        lon2_idx = self.lon_grid_points.index(lon2)

        if data_type == 'netcdf':
            dataset = netCDF4.Dataset(data_path)
        elif data_type == 'bytemap':
            dataset = utils.dataset_of_daily_satel(satel_name,
                                                   data_path)
        vars = dataset.variables

        missing = self.CONFIG[satel_name]['missing_value']

        if satel_name == 'ascat' or satel_name == 'qscat':
            satel_data = self._extract_satel_data_like_ascat(
                satel_name, SatelTable, vars, tc_datetime, data_path,
                lat1_idx, lat2_idx, lon1_idx, lon2_idx, missing)
        elif satel_name == 'wsat':
            satel_data = self._extract_satel_data_like_wsat(
                satel_name, SatelTable, vars, tc_datetime, data_path,
                lat1_idx, lat2_idx, lon1_idx, lon2_idx, missing)
        elif satel_name == 'amsr2':
            satel_data = self._extract_satel_data_like_amsr2(
                satel_name, SatelTable, vars, tc_datetime, data_path,
                lat1_idx, lat2_idx, lon1_idx, lon2_idx, missing)
        elif satel_name == 'smap':
            satel_data = self._extract_satel_data_like_smap(
                satel_name, SatelTable, vars, tc_datetime, data_path,
                lat1_idx, lat2_idx, lon1_idx, lon2_idx, missing)

        if not len(satel_data):
            return

        # Create satellite table
        if sa_table is not None:
            sa_table.create(self.engine)
            self.session.commit()
        # Insert entity into DB
        utils.bulk_insert_avoid_duplicate_unique(
            satel_data,
            int(self.CONFIG['database']['batch_size']['insert']/10),
            SatelTable, ['x_y'], self.session,
            check_self=True)

    def _extract_satel_data_like_smap(self, satel_name, SatelTable,
                                      vars, tc_datetime, data_path,
                                      lat1_idx, lat2_idx,
                                      lon1_idx, lon2_idx, missing):
        satel_data = []

        dataset = netCDF4.Dataset(data_path)
        vars = dataset.variables

        subset = dict()
        var_names = ['minute', 'wind']

        for var_name in var_names:
            subset[var_name] = vars[var_name][lat1_idx:lat2_idx+1,
                                              lon1_idx:lon2_idx+1,
                                              :]

        for y in range(self.grid_2d['lat_axis']):
            for x in range(self.grid_2d['lon_axis']):
                for i in range(self.CONFIG['rss']['passes_number']):

                    if (subset['minute'][y][x][i] is MASKED
                        or subset['wind'][y][x][i] is MASKED):
                        continue

                    # Process the datetime and skip if necessary
                    time_str ='{:02d}{:02d}00'.format(
                        *divmod(int(subset['minute'][y][x][i]), 60))
                    date_str = data_path.split(
                        self.CONFIG['smap']['data_prefix'])[1][:10]\
                            .replace('_', '')

                    if time_str.startswith('24'):
                        date_ = datetime.datetime.strptime(
                            date_str + '000000', '%Y%m%d%H%M%S').date()
                        time_ = datetime.time(0, 0, 0)
                        date_ = date_ + datetime.timedelta(days=1)
                        datetime_ = datetime.datetime.combine(
                            date_, time_)
                    else:
                        datetime_ = datetime.datetime.strptime(
                            date_str + time_str, '%Y%m%d%H%M%S')

                    # Calculate the time delta between satellite datetime
                    # and TC datetime
                    if tc_datetime > datetime_:
                        delta = tc_datetime - datetime_
                    else:
                        delta = datetime_ - tc_datetime
                    # Skip this turn of loop if timedelta is largers than
                    # presetted temporal window
                    if delta.seconds > self.CONFIG['match']['temporal_window']:
                        continue

                    row = SatelTable()
                    row.date_time = datetime_
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[lon1_idx + x]
                    row.lat = self.lat_grid_points[lat1_idx + y]
                    row.x_y = f'{x}_{y}'

                    row.windspd = subset['wind'][y][x][i]

                    # Strictest reading rule: None of columns is none
                    skip = False
                    for key in row.__dict__.keys():
                        if getattr(row, key) is None:
                            skip = True
                            break
                    if skip:
                        continue

                    satel_data.append(row)

        return satel_data


    def _extract_satel_data_like_amsr2(self, satel_name, SatelTable, vars,
                                       tc_datetime, data_path, lat1_idx,
                                       lat2_idx, lon1_idx, lon2_idx,
                                       missing):
        """Extract data of satellite like AMSR2 from bytemap file.

        """
        satel_data = []

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['nodata', 'time', 'sst', 'vapor', 'cloud', 'rain',
                     'land', 'ice', 'windLF', 'windMF']

        for var_name in var_names:
            subset[var_name] = vars[var_name][:,
                                              lat1_idx:lat2_idx+1,
                                              lon1_idx:lon2_idx+1]
        iasc = [0, 1]
        for i in iasc:
            for y in range(self.grid_2d['lat_axis']):
                for x in range(self.grid_2d['lon_axis']):
                    # Skip when the cell has no data
                    if subset['nodata'][i][y][x]:
                        continue

                    # Skip when the time of ascending and descending passes
                    # is same and (w-aw or wdir is not the same)
                    if subset['time'][0][y][x] == vars['time'][1][y][x]:
                        continue

                    if (bool(subset['land'][i][y][x])
                        or bool(subset['ice'][i][y][x])):
                        continue

                    # Process the datetime and skip if necessary
                    time_str ='{:02d}{:02d}00'.format(
                        *divmod(int(60*subset['time'][i][y][x]), 60))
                    data_name = data_path.split('/')[-1]
                    date_str = data_name.split('_')[1][:8]

                    if time_str.startswith('24'):
                        date_ = datetime.datetime.strptime(
                            date_str + '000000', '%Y%m%d%H%M%S').date()
                        time_ = datetime.time(0, 0, 0)
                        date_ = date_ + datetime.timedelta(days=1)
                        datetime_ = datetime.datetime.combine(
                            date_, time_)
                    else:
                        datetime_ = datetime.datetime.strptime(
                            date_str + time_str, '%Y%m%d%H%M%S')

                    # Calculate the time delta between satellite datetime
                    # and TC datetime
                    if tc_datetime > datetime_:
                        delta = tc_datetime - datetime_
                    else:
                        delta = datetime_ - tc_datetime
                    # Skip this turn of loop if timedelta is largers than
                    # presetted temporal window
                    if delta.seconds > self.CONFIG['match']['temporal_window']:
                        continue

                    row = SatelTable()
                    row.date_time = datetime_
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[lon1_idx + x]
                    row.lat = self.lat_grid_points[lat1_idx + y]
                    row.x_y = f'{x}_{y}'

                    row.sst = (None
                               if subset['sst'][i][y][x] == missing
                               else float(subset['sst'][i][y][x]))
                    row.vapor = (None
                                 if subset['vapor'][i][y][x] == missing
                                 else float(subset['vapor'][i][y][x]))
                    row.cloud = (None 
                                 if subset['cloud'][i][y][x] < 0
                                 else float(subset['cloud'][i][y][x]))
                    row.rain = (None 
                                if subset['rain'][i][y][x] < 0
                                else float(subset['rain'][i][y][x]))
                    row.windLF = (None
                                  if subset['windLF'][i][y][x] == missing
                                  else float(subset['windLF'][i][y][x]))
                    row.windMF = (None
                                  if subset['windMF'][i][y][x] == missing
                                  else float(subset['windMF'][i][y][x]))

                    # Strictest reading rule: None of columns is none
                    skip = False
                    for key in row.__dict__.keys():
                        if getattr(row, key) is None:
                            skip = True
                            break
                    if skip:
                        continue

                    satel_data.append(row)

        return satel_data

    def _extract_satel_data_like_wsat(self, satel_name, SatelTable, vars,
                                      tc_datetime, data_path, lat1_idx,
                                      lat2_idx, lon1_idx, lon2_idx,
                                      missing):
        """Extract data of satellite like WindSat from bytemap file.

        """
        satel_data = []

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['nodata', 'mingmt', 'sst', 'vapor', 'cloud', 'rain',
                     'land', 'ice', 'w-lf', 'w-mf', 'w-aw', 'wdir']

        for var_name in var_names:
            subset[var_name] = vars[var_name][:,
                                              lat1_idx:lat2_idx+1,
                                              lon1_idx:lon2_idx+1]
        iasc = [0, 1]
        for i in iasc:
            for y in range(self.grid_2d['lat_axis']):
                for x in range(self.grid_2d['lon_axis']):
                    # Skip when the cell has no data
                    if subset['nodata'][i][y][x]:
                        continue

                    # Skip when the time of ascending and descending passes
                    # is same and (anyone of wind speed variabkles 
                    # or wdir is not the same)
                    if subset['mingmt'][0][y][x] == subset['mingmt'][1][y][x]:
                        skip = False
                        for var_name in ['w-lf', 'w-mf', 'w-aw', 'wdir']:
                            if (subset[var_name][0][y][x] !=\
                                subset[var_name][1][y][x]):
                                skip = True
                        if skip:
                            continue

                    if (bool(subset['land'][i][y][x])
                        or bool(subset['ice'][i][y][x])):
                        continue

                    # Process the datetime and skip if necessary
                    time_str ='{:02d}{:02d}00'.format(
                        *divmod(int(subset['mingmt'][i][y][x]), 60))
                    data_name = data_path.split('/')[-1]
                    date_str = data_name.split('_')[1][:8]

                    if time_str.startswith('24'):
                        date_ = datetime.datetime.strptime(
                            date_str + '000000', '%Y%m%d%H%M%S').date()
                        time_ = datetime.time(0, 0, 0)
                        date_ = date_ + datetime.timedelta(days=1)
                        datetime_ = datetime.datetime.combine(
                            date_, time_)
                    else:
                        datetime_ = datetime.datetime.strptime(
                            date_str + time_str, '%Y%m%d%H%M%S')

                    # Calculate the time delta between satellite datetime
                    # and TC datetime
                    if tc_datetime > datetime_:
                        delta = tc_datetime - datetime_
                    else:
                        delta = datetime_ - tc_datetime
                    # Skip this turn of loop if timedelta is largers than
                    # presetted temporal window
                    if delta.seconds > self.CONFIG['match']['temporal_window']:
                        continue

                    row = SatelTable()
                    row.date_time = datetime_
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[lon1_idx + x]
                    row.lat = self.lat_grid_points[lat1_idx + y]
                    row.x_y = f'{x}_{y}'

                    row.sst = (None
                               if subset['sst'][i][y][x] == missing
                               else float(subset['sst'][i][y][x]))
                    row.vapor = (None
                                 if subset['vapor'][i][y][x] == missing
                                 else float(subset['vapor'][i][y][x]))
                    row.cloud = (None 
                                 if subset['cloud'][i][y][x] < 0
                                 else float(subset['cloud'][i][y][x]))
                    row.rain = (None 
                                if subset['rain'][i][y][x] < 0
                                else float(subset['rain'][i][y][x]))

                    row.w_lf = (None
                                if subset['w-lf'][i][y][x] == missing
                                else float(subset['w-lf'][i][y][x]))
                    row.w_mf = (None
                                if subset['w-mf'][i][y][x] == missing
                                else float(subset['w-mf'][i][y][x]))
                    row.w_aw = (None
                                if subset['w-aw'][i][y][x] == missing
                                else float(subset['w-aw'][i][y][x]))
                    row.winddir = (None
                                   if subset['wdir'][i][y][x] == missing
                                   else float(subset['wdir'][i][y][x]))

                    # Strictest reading rule: None of columns is none
                    skip = False
                    for key in row.__dict__.keys():
                        if getattr(row, key) is None:
                            skip = True
                            break
                    if skip:
                        continue

                    satel_data.append(row)

        return satel_data

    def _extract_satel_data_like_ascat(self, satel_name, SatelTable, vars,
                                       tc_datetime, data_path, lat1_idx,
                                       lat2_idx, lon1_idx, lon2_idx,
                                       missing):
        """
        Extract data of satellites like ASCAT from bytemap file.

        Notes
        -----
        Unlike QuikSCAT, ASCAT wind retrievals in rain at high winds,
        such as tropical storms, are quite good.

        """
        satel_data = []

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['nodata', 'mingmt', 'scatflag', 'radrain', 'sos',
                     'land', 'ice', 'windspd', 'winddir']
        if satel_name == 'qscat':
            var_names.remove('sos')

        for var_name in var_names:
            subset[var_name] = vars[var_name][:,
                                              lat1_idx:lat2_idx+1,
                                              lon1_idx:lon2_idx+1]
        iasc = [0, 1]
        for i in iasc:
            for y in range(self.grid_2d['lat_axis']):
                for x in range(self.grid_2d['lon_axis']):
                    # Skip when the cell has no data
                    if subset['nodata'][i][y][x]:
                        continue

                    # Skip when the time of ascending and descending passes
                    # is same
                    if subset['mingmt'][0][y][x] == subset['mingmt'][1][y][x]:
                        continue

                    if (bool(subset['land'][i][y][x])
                        or bool(subset['ice'][i][y][x])):
                        continue

                    # In ASCAT document, it said that 'We suggest
                    # discarding observations for which SOSAL.GT.1.9.'
                    if (satel_name == 'ascat' and
                        float(subset['sos'][i][y][x]) > 1.9):
                        continue

                    # Process the datetime and skip if necessary
                    time_str ='{:02d}{:02d}00'.format(
                        *divmod(int(subset['mingmt'][i][y][x]), 60))
                    data_name = data_path.split('/')[-1]
                    date_str = data_name.split('_')[1][:8]

                    if time_str.startswith('24'):
                        date_ = datetime.datetime.strptime(
                            date_str + '000000', '%Y%m%d%H%M%S').date()
                        time_ = datetime.time(0, 0, 0)
                        date_ = date_ + datetime.timedelta(days=1)
                        datetime_ = datetime.datetime.combine(
                            date_, time_)
                    else:
                        datetime_ = datetime.datetime.strptime(
                            date_str + time_str, '%Y%m%d%H%M%S')

                    # Calculate the time delta between satellite datetime
                    # and TC datetime
                    if tc_datetime > datetime_:
                        delta = tc_datetime - datetime_
                    else:
                        delta = datetime_ - tc_datetime
                    # Skip this turn of loop if timedelta is largers than
                    # presetted temporal window
                    if delta.seconds > self.CONFIG['match']['temporal_window']:
                        continue

                    row = SatelTable()
                    row.date_time = datetime_
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[lon1_idx + x]
                    row.lat = self.lat_grid_points[lat1_idx + y]
                    row.x_y = f'{x}_{y}'
                    row.scatflag = (None
                                    if subset['scatflag'][i][y][x] == missing
                                    else bool(subset['scatflag'][i][y][x]))
                    row.radrain = (None 
                                   if subset['radrain'][i][y][x] < 0
                                   else float(subset['radrain'][i][y][x]))

                    row.windspd = (None
                                   if subset['windspd'][i][y][x] == missing
                                   else float(subset['windspd'][i][y][x]))

                    row.winddir = (None
                                   if subset['winddir'][i][y][x] == missing
                                   else float(subset['winddir'][i][y][x]))

                    # Strictest reading rule: None of columns is none
                    skip = False
                    for key in row.__dict__.keys():
                        if getattr(row, key) is None:
                            skip = True
                            break
                    if skip:
                        continue

                    satel_data.append(row)

        return satel_data

    def download(self, satel_name, target_datetime):
        """Download satellite on specified date of TC.

        """
        if not self._datetime_in_satel_lifetime(satel_name,
                                                target_datetime):
            return None

        satel_config = self.CONFIG[satel_name]
        satel_date = target_datetime.date()
        file_path = self._download_satel_data_in_specified_date(
            satel_config, satel_name, satel_date)

        return file_path

    def _datetime_in_satel_lifetime(self, satel_name, tc_datetime):
        """Check whether TC datetime is in the range of satellite's
        lifetime.

        """
        period_limit = self.CONFIG[satel_name]['period_limit']
        if (tc_datetime >= period_limit['start']
            and tc_datetime <= period_limit['end']):
            return True

        return False

    def _download_satel_data_in_specified_date(
        self, config, satel_name, satel_date):
        """Download ASCAT/QucikSCAT/Windsat data on specified date.

        """
        self.logger.debug(f'Downloading {satel_name} data on {satel_date}')

        data_url = config['urls']
        file_prefix = config['data_prefix']
        file_suffix = config['data_suffix']
        save_dir = config['dirs']['bmaps']
        missing_dates_file = config['files_path']['missing_dates']

        if os.path.exists(missing_dates_file):
            with open(missing_dates_file, 'rb') as fr:
                missing_dates = pickle.load(fr)
        else:
            missing_dates = set()

        utils.set_format_custom_text(config['data_name_length'])
        os.makedirs(save_dir, exist_ok=True)

        if satel_date in missing_dates:
            return None
        if satel_name == 'smap':
            file_name = '%s%04d_%02d_%02d%s' % (
                file_prefix, satel_date.year, satel_date.month,
                satel_date.day, file_suffix)
            file_url = f'{data_url}{satel_date.year}/{file_name}'
        else:
            file_name = '%s_%04d%02d%02d%s' % (
                file_prefix, satel_date.year, satel_date.month,
                satel_date.day, file_suffix)
            file_url = '%sy%04d/m%02d/%s' % (
                data_url, satel_date.year, satel_date.month, file_name)

        if not utils.url_exists(file_url):
            print('Missing date of {satel_name}: {satel_date}')
            print(file_url)
            missing_dates.add(satel_date)
            return None

        file_path = f'{save_dir}{file_name}'

        utils.download(file_url, file_path)

        with open(missing_dates_file, 'wb') as fw:
            pickle.dump(missing_dates, fw)

        return file_path
