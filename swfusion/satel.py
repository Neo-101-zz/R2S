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

import utils
import cwind
import sfmr

Base = declarative_base()

class SatelManager(object):
    """Manage features of satellite data that are not related to other data
    sources except TC table from IBTrACS.

    """

    def __init__(self, CONFIG, period, region, passwd,
                 spatial_window, temporal_window):
        self.logger = logging.getLogger(__name__)
        self.satel_names = ['amsr2', 'ascat', 'qscat', 'wsat']
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
        self.lat_grid_points = [y * self.spa_resolu - 89.875 for y in range(
            self.CONFIG['rss']['lat_grid_points_number'])]
        self.lon_grid_points = [x * self.spa_resolu + 0.125 for x in range(
            self.CONFIG['rss']['lon_grid_points_number'])]

        # Size of 3D grid points around TC center
        self.grid_2d = dict()
        self.grid_2d['lat_axis'] = self.grid_2d['lon_axis'] = int(
            self.edge/self.spa_resolu)
        self.missing_value = self.CONFIG['rss']['missing_value']

        utils.setup_database(self, Base)
        self.download_and_read()

    def _get_basic_columns(self):
        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('x_y', String(20), nullable=False,
                           unique=True))

        return cols

    def _get_satel_table_class(self, satel_name, sid, dt,
                               lat_index, lon_index):
        dt_str = dt.strftime('%Y_%m%d_%H%M')
        table_name = f'{satel_name}_tc_{sid}_{dt_str}_{lon_index}_{lat_index}'

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return table_name, None, Satel

        cols = self._get_basic_columns()
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

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        mapper(Satel, t)

        return table_name, t, Satel

    def download_and_read(self):
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

            # Get index of latitude and longitude of closest RSS cell
            # center point near TC center
            lat_index, lon_index = \
                    utils.get_latlon_index_of_closest_grib_point(
                        row.lat, row.lon, self.lat_grid_points,
                        self.lon_grid_points)

            for satel_name in self.satel_names:

                # Get satellite table
                table_name, sa_table, SatelTable = \
                        self._get_satel_table_class(
                            satel_name, row.sid, tc_datetime,
                            lat_index, lon_index)
                # Download satellite data according to TC datetime
                bytemap_path = self.download(satel_name, tc_datetime)
                utils.reset_signal_handler()

                if bytemap_path is None:
                    continue

                # Show the TC area and sateliite data area in the
                # temporal window
                hit_count = self.show_match(lat1, lat2, lon1, lon2,
                                            satel_name, row, bytemap_path,
                                            draw=False)
                if not hit_count[self.edge]:
                    continue

                # Read satellite data according to TC datetime
                self.read(satel_name, bytemap_path, tc_datetime,
                          table_name, sa_table, SatelTable,
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

    def get_satel_coverage(self, tc_row, satel_name, bm_file, draw):
        dataset = utils.dataset_of_daily_satel(satel_name, bm_file)
        vars = dataset.variables

        bm_file_name = bm_file.split('/')[-1]
        date_str = bm_file_name.split('_')[1][:8]

        lats = []
        lons = []

        for i in range(self.CONFIG['rss']['passes_number']):
            for j in range(self.CONFIG['rss']['lat_grid_points_number']):
                for k in range(self.CONFIG['rss']['lon_grid_points_number']):

                    if not self.check_match(vars, i, j, k,
                                            tc_row.date_time, date_str):
                        continue
                    lats.append(vars['latitude'][j])
                    lons.append(vars['longitude'][k])

                    if draw:
                        self.draw_coverage_within_window(vars, j, k)

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

            print((f'{satel_name} {tc_row.sid} {tc_row.name} '
                   + f'{tc_row.date_time} {e}: {count[e]}'))

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

    def draw_coverage_within_window(self, vars, j, k):
        lat = vars['latitude'][j]
        lon = vars['longitude'][k] - 180

        grid_half_edge = (0.5 * self.CONFIG['rss']\
                          ['spatial_resolution'])

        lat1, lat2 = lat - grid_half_edge, lat + grid_half_edge
        lon1, lon2 = lon - grid_half_edge, lon + grid_half_edge
        lats = np.array([lat1, lat2, lat2, lat1])
        lons = np.array([lon1, lon1, lon2, lon2])
        self.draw_cloud_pixel(lats, lons, map, 'white')

    def check_match(self, vars, i, j, k, tc_datetime, date_str):
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
                   tc_row, bm_file, draw):

        map = Basemap(llcrnrlon=-180.0, llcrnrlat=-90.0, urcrnrlon=180.0,
                      urcrnrlat=90.0)
        map.drawcoastlines()
        map.drawmapboundary(fill_color='aqua')
        map.fillcontinents(color='coral', lake_color='aqua')
        map.drawmeridians(np.arange(0, 360, 30))
        map.drawparallels(np.arange(-90, 90, 30))

        self.draw_tc_area(map, lat1, lat2, lon1, lon2, 10)

        count = self.get_satel_coverage(tc_row, satel_name, bm_file,
                                        draw)
        if not draw:
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

    def read(self, satel_name, bytemap_path, tc_datetime, table_name,
             sa_table, SatelTable, lat1, lat2, lon1, lon2):
        self.logger.debug(f'Reading {satel_name}: {bytemap_path}')

        lat1_idx = self.lat_grid_points.index(lat1)
        lat2_idx = self.lat_grid_points.index(lat2)
        lon1_idx = self.lon_grid_points.index(lon1)
        lon2_idx = self.lon_grid_points.index(lon2)

        dataset = utils.dataset_of_daily_satel(satel_name, bytemap_path)
        vars = dataset.variables

        missing = self.CONFIG[satel_name]['missing_value']

        if satel_name == 'ascat' or satel_name == 'qscat':
            satel_data = self._extract_satel_data_like_ascat(
                satel_name, SatelTable, vars, tc_datetime, bytemap_path,
                lat1_idx, lat2_idx, lon1_idx, lon2_idx, missing)
        elif satel_name == 'wsat':
            satel_data = self._extract_satel_data_like_wsat(
                satel_name, SatelTable, vars, tc_datetime, bytemap_path,
                lat1_idx, lat2_idx, lon1_idx, lon2_idx, missing)
        elif satel_name == 'amsr2':
            satel_data = self._extract_satel_data_like_amsr2(
                satel_name, SatelTable, vars, tc_datetime, bytemap_path,
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

    def _extract_satel_data_like_amsr2(self, satel_name, SatelTable, vars,
                                       tc_datetime, bm_file, lat1_idx,
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
                    bm_file_name = bm_file.split('/')[-1]
                    date_str = bm_file_name.split('_')[1][:8]

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
                                      tc_datetime, bm_file, lat1_idx,
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
                    bm_file_name = bm_file.split('/')[-1]
                    date_str = bm_file_name.split('_')[1][:8]

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
                                       tc_datetime, bm_file, lat1_idx,
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
                    bm_file_name = bm_file.split('/')[-1]
                    date_str = bm_file_name.split('_')[1][:8]

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

    def download(self, satel_name, tc_datetime):
        """Download satellite on specified date of TC.

        """
        # self.logger.info((f'Downloading {satel_name} data on '
        #                   + f'{tc_datetime.date()}'))

        if not self._datetime_in_satel_lifetime(satel_name, tc_datetime):
            return None

        satel_config = self.CONFIG[satel_name]
        satel_date = tc_datetime.date()
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
