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

from ascat_daily import ASCATDaily
from quikscat_daily_v4 import QuikScatDaily
from windsat_daily_v7 import WindSatDaily
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
        self.satel_names = ['ascat', 'qscat', 'wsat']
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

        self.edge = self.CONFIG['era5']['subset_edge_in_degree']
        self.spa_resolu = self.CONFIG['rss']['spatial_resolution']
        self.lat_grid_points = [y * self.spa_resolu - 89.875 for y in range(
            self.CONFIG['era5']['lat_grid_points_number'])]
        self.lon_grid_points = [x * self.spa_resolu + 0.125 for x in range(
            self.CONFIG['era5']['lon_grid_points_number'])]

        # Size of 3D grid points around TC center
        self.grid_2d = dict()
        self.grid_2d['lat_axis'] = self.grid_2d['lon_axis'] = int(
            self.edge/self.spa_resolu)
        self.missing_value = self.CONFIG['rss']['missing_value']

        # self.download()

        utils.reset_signal_handler()

        utils.setup_database(self, Base)
        self.read(read_all=True)
        # self.show()
        # self.match()

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
        dt_str = dt.strftime('%Y%m%d%H%M%S')
        table_name = f'{satel_name}_tc_{sid}_{dt_str}_{lat_index}_{lon_index}'

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return table_name, None, Satel

        cols = self._get_basic_columns()
        cols.append(Column('windspd', Float, nullable=False))
        cols.append(Column('winddir', Float, nullable=False))
        if satel_name == 'ascat' or satel_name == 'qscat':
            cols.append(Column('scatflag', Boolean, nullable=False))
            cols.append(Column('radrain', Float, nullable=False))
        elif satel_name == 'wsat':
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
        info = f'Downloading satellite data according to TC records'
        self.logger.info(info)

        # Loop all TC records
        for row in self.session.query(TCTable).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            # Get TC datetime
            tc_datetime = row.datetime

            # Get hit result and range of satellite data matrix near
            # TC center
            hit, lat1, lat2, lon1, lon2 = \
                    utils.get_subset_range_of_grib(
                        row.lat, row.lon, self.lat_grid_points,
                        self.lon_grid_points, self.edge)
            if not hit:
                continue

            count += 1
            print(f'\r{info} {count}/{total}', end='')

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
                if bytemap_path is None:
                    continue
                # Read satellite data according to TC datetime
                self.read(satel_name, bytemap_path, tc_datetime,
                          table_name, sa_table, SatelTable,
                          lat1, lat2, lon1, lon2)

    def read(self, satel_name, bytemap_path, tc_datetime, table_name,
             sa_table, SatelTable, lat1, lat2, lon1, lon2):
        self.logger.info(f'Reading {satel_name}: {bytemap_path}')
        if satel_name == 'ascat' or satel_name == 'qscat':
            satel_data = self._extract_satel_data_like_ascat(
                satel_name, SatelTable, file_path, tc_datetime,
                lat1, lat2, lon1, lon2)
        elif satel_name == 'wsat':
            satel_data = self._extract_satel_data_like_wsat(
                self, SatelTable, file_path, tc_datetime,
                lat1, lat2, lon1, lon2)
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
            SatelTable, ['x_y_z'], self.session,
            check_self=True)

    def _extract_satel_data_like_wsat(self, SatelTable, file_path,
                                       tc_datetime, lat1, lat2, lon1, lon2):
        """Extract data of satellite like WindSat from bytemap file.

        """
        satel_data = []

        lat1_idx = self.lat_grid_points.index(lat1)
        lat2_idx = self.lat_grid_points.index(lat2)
        lon1_idx = self.lon_grid_points.index(lon1)
        lon2_idx = self.lon_grid_points.index(lon2)

        dataset = utils.dataset_of_daily_satel(satel_name, file_path)
        vars = dataset.variables

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['nodata', 'mingmt', 'sst', 'vapor', 'cloud', 'rain',
                     'land', 'ice', 'w-aw', 'wdir']

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
                    if (subset['minmgt'][0][y][x] == mingmt[1][y][x]
                        and (subset['w-aw'][0][y][x] !=\
                             subset['w-aw'][1][y][x]
                             or subset['wdir'][0][y][x] !=\
                             subset['wdir'][1][y][x])
                       ):
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
                    if tc_datetime != datetime_:
                        continue

                    row = SatelTable()
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[lon1_index + x]
                    row.lat = self.lat_grid_points[lat1_index + y]
                    row.x_y = f'{x}_{y}'

                    row.sst= (None
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
                    row.windspd = float(subset['w-aw'][i][y][x])
                    row.winddir = float(subset['wdir'][i][y][x])

                    satel_data.append(row)

        return satel_data

    def _extract_satel_data_like_ascat(self, satel_name, SatelTable,
                                       file_path, tc_datetime,
                                       lat1, lat2, lon1, lon2, missing):
        """
        Extract data of satellites like ASCAT from bytemap file.

        Notes
        -----
        Unlike QuikSCAT, ASCAT wind retrievals in rain at high winds,
        such as tropical storms, are quite good.

        """
        satel_data = []

        lat1_idx = self.lat_grid_points.index(lat1)
        lat2_idx = self.lat_grid_points.index(lat2)
        lon1_idx = self.lon_grid_points.index(lon1)
        lon2_idx = self.lon_grid_points.index(lon2)

        dataset = utils.dataset_of_daily_satel(satel_name, file_path)
        vars = dataset.variables

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
                    if subset['minmgt'][0][y][x] == mingmt[1][y][x]:
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
                    if tc_datetime != datetime_:
                        continue

                    row = SatelTable()
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[lon1_index + x]
                    row.lat = self.lat_grid_points[lat1_index + y]
                    row.x_y = f'{x}_{y}'
                    row.scatflag = (None
                                    if subset['scatflag'][i][y][x] == missing
                                    else bool(subset['scatflag'][i][y][x]))
                    row.radrain = (None 
                                   if subset['radrain'][i][y][x] < 0
                                   else float(subset['radrain'][i][y][x]))
                    row.windspd = float(subset['windspd'][i][y][x])
                    row.winddir = float(subset['winddir'][i][y][x])

                    satel_data.append(row)

        return satel_data

    def download(self, satel_name, tc_datetime):
        """Download satellite on specified date of TC.

        """
        self.logger.info(f'Downloading {satel_name} data on {tc_datetime}')

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
        info = config['prompt']['info']['download']
        self.logger.info(info)

        data_url = config['urls']
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
            satel_name, satel_date.year, satel_date.month,
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
