"""Manage downloading and reading ASCAT, QucikSCAT and Windsat data.

"""
import datetime
import math
import logging
import pickle
import operator
import os
import time
import sys

from geopy import distance
import mysql.connector
from sqlalchemy import create_engine, extract
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

class ASCATManager(object):
    """Manage features of satellite data that are not related to other data
    sources except TC table from IBTrACS.

    """

    def __init__(self, CONFIG, period, region, passwd, save_disk):
        self.logger = logging.getLogger(__name__)
        self.satel_names = ['ascat']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.save_disk = save_disk

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
        self._get_region_corners_indices()
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
            self.edge/self.spa_resolu) + 1

        self.missing_value = dict()
        for satel_name in self.satel_names:
            if satel_name != 'smap':
                self.missing_value[satel_name] = \
                        self.CONFIG[satel_name]['missing_value']
            else:
                self.missing_value[satel_name] = dict()
                self.missing_value[satel_name]['minute'] = \
                        self.CONFIG[satel_name]['missing_value']['minute']
                self.missing_value[satel_name]['wind'] = \
                        self.CONFIG[satel_name]['missing_value']['wind']

        utils.setup_database(self, Base)
        # self.download_and_read_tc_oriented()
        self.download_and_read()

    def _get_region_corners_indices(self):
        self.lat1_index = self.lat_grid_points.index(self.lat1 - 0.5 *
                                                     self.spa_resolu)
        self.lat2_index = self.lat_grid_points.index(self.lat2 + 0.5 *
                                                     self.spa_resolu)
        self.lon1_index = self.lon_grid_points.index(self.lon1 - 0.5 *
                                                     self.spa_resolu)
        self.lon2_index = self.lon_grid_points.index(self.lon2 + 0.5 *
                                                     self.spa_resolu)

    def get_satel_era5_table(self, satel_name, target_datetime):
        if satel_name == 'ascat':
            table_name, t, SatelERA5 = self.get_ascat_era5_table(
                target_datetime)

        return table_name, t, SatelERA5

    def get_ascat_era5_table(self, dt):
        table_name = f'ascat_{dt.year}_{dt.month}'

        class ASCAT(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(ASCAT, t)

            return table_name, None, ASCAT

        cols = utils.get_basic_satel_era5_columns()
        cols.append(Column('windspd', Float, nullable=False))
        cols.append(Column('winddir', Float, nullable=False))
        cols.append(Column('scatflag', Boolean, nullable=False))
        cols.append(Column('radrain', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, work=False,
                                 save_disk=self.save_disk)
        era5_cols = era5_.get_era5_columns()
        cols = satel_cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        mapper(ASCAT, t)

        return table_name, t, ASCAT

    def download_and_read(self):

        utils.setup_signal_handler()

        dt_delta = self.period[1] - self.period[0]

        for day in range(dt_delta.days):
            target_datetime = (self.period[0]
                               + datetime.timedelta(days=day))
            self.logger.info((f'Download and reading satel and ERA5 data: '
                              + f'{target_datetime.date()}'))

            era5_ = era5.ERA5Manager(self.CONFIG, self.period,
                                     self.region,
                                     self.db_root_passwd,
                                     work=False,
                                     save_disk=self.save_disk)
            era5_data_path = \
                    era5_.download_all_surface_vars_of_whole_day(
                        target_datetime)
            for satel_name in self.satel_names:
                # Download satellite data
                satel_data_path = self.download(satel_name, target_datetime)
                utils.reset_signal_handler()
                if satel_data_path is None:
                    continue
                # Download ERA5 data
                self.logger.debug((f'Downloading all surface ERA5 '
                                   + f'data of all hours on '
                                   + f'{target_datetime.date()}'))

                # Get satellite table
                SatelERA5 = self.get_satel_era5_table(satel_name, target_datetime)

                self.logger.debug((f'Reading {satel_name} and ERA5 '
                                   + f'data on '
                                   + f'{target_datetime.date()}'))
                self.read_satel_era5(satel_name, satel_data_path,
                                     era5_data_path, SatelERA5,
                                     target_datetime.date())
            if self.save_disk:
                os.remove(era5_data_path)
            breakpoint()

    def read_satel_era5(self, satel_name, satel_data_path,
                        era5_data_path, SatelERA5i, target_date):
        # Extract satel and ERA5 data
        satel_era5_data = self.extract_satel_era5(
            satal_name, satel_data_path, era5_data_path, SatelERA5,
            target_date)
        if satel_era5_data is None:
            return

        # Insert into table
        utils.bulk_insert_avoid_duplicate_unique(
            satel_era5_data, self.CONFIG['database']\
            ['batch_size']['insert'],
            SatelERA5, ['satel_datetime_lon_lat'], self.session,
            check_self=True)

    def extract_satel_era5(self, satel_name, satel_data_path,
                           era5_data_path, SatelERA5, target_date):
        # Get SMAP part first
        self.logger.debug(f'Getting SMAP part from {satel_data_path}')
        satel_part = self._get_satel_part(satel_name, satel_data_path,
                                          SatelERA5, target_date)
        if not len(satel_part):
            return None

        # Then add ERA5 part
        self.logger.debug(f'Adding ERA5 part from {era5_data_path}')
        satel_era5_data = self._add_era5_part(satel_part, era5_data_path)

        return satel_era5_data

    def _get_satel_part(self, satel_name, satel_data_path, SatelERA5,
                        target_date):
        if satel_name == 'ascat':
            satel_part = self._get_ascat_part(satel_data_path, SatelERA5,
                                              target_date)

        return satel_part

    def _add_era5_part(self, satel_part, era5_data_path):
        """
        Attention
        ---------
        Break this function into several sub-functions is not
        recommended. Because it needs to deliver large arguments
        such as satel_part and grib message.

        """
        satel_era5_data = []
        grbidx = pygrib.index(era5_data_path, 'dataTime')
        count = 0

        info = f'Adding ERA5 from {era5_data_path}: '
        total = len(satel_part) * 16
        # Get temporal relation of grbs and rows first
        hourtime_row = self._get_hourtime_row_dict(satel_part)
        grb_date_str = satel_part[0].satel_datetime.strftime(
            '%Y%m%d')

        # For every hour, update corresponding rows with grbs
        for hourtime in range(0, 2400, 100):
            if not len(hourtime_row[hourtime]):
                continue

            grb_time = hourtime
            if grb_time == 0:
                grb_time = '000'
            grb_datetime = datetime.datetime.strptime(
                f'{grb_date_str}{grb_time}', '%Y%m%d%H%M%S')

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

                    row = satel_part[idx]

                    row.era5_datetime = grb_datetime
                    satel_minute = (row.satel_datetime.hour * 60
                                     + row.satel_datetime.minute)
                    grb_minute = int(hourtime/100) * 60
                    row.satel_era5_diff_mins = \
                            satel_minute - grb_minute

                    lat1, lat2, lon1, lon2 = self._get_corners_of_cell(
                        row.lon, row.lat)
                    if lon2 == 360:
                        lon2 = 0
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

        return satel_part

    def _get_ascat_part(self, satel_data_path, SatelERA5, target_date):
        satel_data = []

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['nodata', 'mingmt', 'scatflag', 'radrain', 'sos',
                     'land', 'ice', 'windspd', 'winddir']

        for var_name in var_names:
            subset[var_name] = vars[var_name][
                :,
                self.lat1_index: self.lat2_index+1,
                self.lon1_index: self.lon2_index+1]

        passes_num, lats_num, lons_num = subset[var_names[0]].shape

        for i in range(passes_num):
            for y in range(lats_num):
                for x in range(lons_num):
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
                    if float(subset['sos'][i][y][x]) > 1.9:
                        continue

                    # Process the datetime and skip if necessary
                    time_ = datetime.time(
                        *divmod(int(subset['mingmt'][i][y][x]),60), 0)

                    row = SatelTable()
                    row.satel_datetime = datetime.datetime.combine(
                        target_date, time_)
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[self.lon1_index + x]
                    row.lat = self.lat_grid_points[self.lat1_index + y]
                    row.satel_datetime_x_y = f'{row.satel_datetime}_{x}_{y}'
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


