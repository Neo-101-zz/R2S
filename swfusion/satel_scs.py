"""Manage downloading and reading ASCAT, QucikSCAT and Windsat data.

"""
import datetime
import math
import logging
import pickle
import operator
import random
import re
import os
import signal
import time
import sys
import xml.etree.ElementTree as ET
import zipfile
from statistics import mean
from multiprocessing import Pool

from geopy import distance
import requests
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
import pandas as pd
import scipy

import utils
import cwind
import sfmr
import era5
import grid

MASKED = np.ma.core.masked
Base = declarative_base()
current_file = None

class SCSSatelManager(object):
    """Manage features of satellite data that are not related to other data
    sources except TC table from IBTrACS.

    """

    def __init__(self, CONFIG, period, region, passwd, save_disk):
        self.logger = logging.getLogger(__name__)
        # self.satel_names = ['ascat', 'wsat', 'amsr2', 'smap']
        # self.satel_names = ['sentinel_1']
        self.satel_names = ['sentinel_1']

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

        self.spa_resolu = dict()
        self.spa_resolu['rss'] = self.CONFIG['rss']['spatial_resolution']
        self.spa_resolu['grid'] = self.CONFIG['grid']\
                ['spatial_resolution']

        self.grid_pts = dict()

        self.grid_pts['rss'] = dict()
        self.grid_pts['rss']['lat'] = [
            y * self.spa_resolu['rss'] - 89.875 for y in range(
                self.CONFIG['rss']['lat_grid_points_number'])
        ]
        self.grid_pts['rss']['lon'] = [
            x * self.spa_resolu['rss'] + 0.125 for x in range(
                self.CONFIG['rss']['lon_grid_points_number'])
        ]

        self.grid_pts['era5'] = dict()
        self.grid_pts['era5']['lat'] = [
            y * self.spa_resolu['rss'] - 90 for y in range(
                self.CONFIG['era5']['lat_grid_points_number'])
        ]
        self.grid_pts['era5']['lon'] = [
            x * self.spa_resolu['rss'] for x in range(
                self.CONFIG['era5']['lon_grid_points_number'])
        ]

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

        self._get_region_corners_indices()
        utils.setup_database(self, Base)

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        # self.download_and_read_tc_oriented()
        self.download_and_read()

    def _get_region_corners_indices(self):
        self.lat1_index = self.grid_pts['rss']['lat'].index(
            self.lat1 + 0.5 * self.spa_resolu['rss'])
        self.lat2_index = self.grid_pts['rss']['lat'].index(
            self.lat2 - 0.5 * self.spa_resolu['rss'])
        self.lon1_index = self.grid_pts['rss']['lon'].index(
            self.lon1 + 0.5 * self.spa_resolu['rss'])
        self.lon2_index = self.grid_pts['rss']['lon'].index(
            self.lon2 - 0.5 * self.spa_resolu['rss'])

    def create_satel_era5_table(self, satel_name, target_datetime):
        if satel_name == 'ascat':
            SatelERA5 = self.create_ascat_era5_table(target_datetime)
        elif satel_name == 'wsat':
            SatelERA5 = self.create_wsat_era5_table(target_datetime)
        elif satel_name == 'amsr2':
            SatelERA5 = self.create_amsr2_era5_table(target_datetime)
        elif satel_name == 'smap':
            SatelERA5 = self.create_smap_era5_table(target_datetime)
        elif satel_name == 'sentinel_1':
            SatelERA5 = self.create_sentinel_1_table(target_datetime)

        return SatelERA5

    def download_and_read(self):

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
            era5_data_path = dict()
            self.logger.info((f'Downloading ERA5 data '
                              + f'on {target_datetime.date()}'))
            era5_data_path['today'] = \
                    era5_.download_all_surface_vars_of_whole_day(
                        target_datetime)

            tomorrow = target_datetime + datetime.timedelta(days=1)
            self.logger.info((f'Downloading ERA5 data '
                              + f'on {tomorrow.date()}'))
            era5_data_path['tomorrow'] = \
                    era5_.download_all_surface_vars_of_whole_day(
                        tomorrow)

            for satel_name in self.satel_names:
                # Download satellite data
                satel_data_path = self.download(satel_name,
                                                target_datetime)
                if satel_data_path is None:
                    continue

                # Get satellite table
                SatelERA5 = self.create_satel_era5_table(
                    satel_name, target_datetime)
                self.logger.debug((f'Reading {satel_name} and ERA5 '
                                   + f'data on '
                                   + f'{target_datetime.date()}'))
                self.read_satel_era5(satel_name, satel_data_path,
                                     era5_data_path, SatelERA5,
                                     target_datetime.date())

            if self.save_disk:
                os.remove(era5_data_path['today'])
            breakpoint()

    def get_grid_pts_in_range(self, min_lon, max_lon, min_lat, max_lat):
        Grid = grid.Grid

        pts = self.session.query(Grid).filter(
            Grid.lon > min_lon, Grid.lon < max_lon,
            Grid.lat > min_lat, Grid.lat < max_lat,
            Grid.land == False)

        return pts

    def read_satel_era5(self, satel_name, satel_data_path,
                        era5_data_path, SatelERA5, target_date):
        # Extract satel and ERA5 data
        satel_era5_data = self.extract_satel_era5(
            satel_name, satel_data_path, era5_data_path, SatelERA5,
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
        self.logger.debug(f'Getting satellite part from {satel_data_path}')
        satel_part = self._get_satel_part(satel_name, satel_data_path,
                                          SatelERA5, target_date)
        if not len(satel_part):
            return None

        # Then add ERA5 part
        self.logger.debug(f'Adding ERA5 part from {era5_data_path}')
        satel_era5_data = self._add_era5_part(
            satel_name, satel_part, era5_data_path, target_date)

        return satel_era5_data

    def _get_satel_part(self, satel_name, satel_data_path, SatelERA5,
                        target_date):
        info = f'Getting {satel_name} part'
        if satel_name != 'sentinel_1':
            self.logger.info(f'{info} from {satel_data_path}')
        else:
            self.logger.info(info)

        missing = self.CONFIG[satel_name]['missing_value']

        if satel_name != 'smap' and satel_name != 'sentinel_1':
            dataset = utils.dataset_of_daily_satel(satel_name,
                                                   satel_data_path)
            vars = dataset.variables

            if satel_name == 'ascat':
                satel_part = self._get_ascat_part(vars, SatelERA5,
                                                  target_date, missing)
            elif satel_name == 'wsat':
                satel_part = self._get_wsat_part(vars, SatelERA5,
                                                 target_date, missing)
            elif satel_name == 'amsr2':
                satel_part = self._get_amsr2_part(vars, SatelERA5,
                                                 target_date, missing)
        elif satel_name == 'smap':
            dataset = netCDF4.Dataset(satel_data_path)

            # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which
            # faster than 1 m/s, so must disable auto mask
            dataset.set_auto_mask(False)
            vars = dataset.variables

            satel_part = self._get_smap_part(vars, SatelERA5,
                                             target_date, missing)
        elif satel_name == 'sentinel_1':
            satel_part = self._get_sentinel_1_part(
                satel_data_path, SatelERA5, target_date)

        return satel_part

    def _add_era5_part(self, satel_name, satel_part, era5_data_path,
                       target_date):
        """
        Attention
        ---------
        Break this function into several sub-functions is not
        recommended. Because it needs to deliver large arguments
        such as satel_part and grib message.

        """
        satel_era5_data = []
        total = len(satel_part) * 16
        count = 0
        # Get temporal relation of grbs and rows first
        hourtime_row, next_day_indices = self._get_hourtime_row_dict(
            satel_part)

        for which_day in ['today', 'tomorrow']:
            info = f'Adding ERA5 from {era5_data_path[which_day]}: '
            grbidx = pygrib.index(era5_data_path[which_day], 'dataTime')

            # For every hour, update corresponding rows with grbs
            for hourtime in range(0, 2400, 100):
                if not len(hourtime_row[hourtime]):
                    continue
                grb_time = datetime.time(int(hourtime/100), 0, 0)

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
                        if (which_day == 'today'
                            and idx in next_day_indices):
                            continue
                        if (which_day == 'tomorrow'
                            and idx not in next_day_indices):
                            continue

                        grb_datetime = datetime.datetime.combine(
                            target_date, grb_time)
                        if which_day == 'tomorrow':
                            grb_datetime = (grb_datetime
                                            + datetime.timedelta(days=1))

                        count += 1
                        print(f'\r{info} {count}/{total}', end='')

                        row = satel_part[idx]

                        row.era5_datetime = grb_datetime
                        satel_minute = (row.satel_datetime.hour * 60
                                         + row.satel_datetime.minute)
                        grb_minute = int(hourtime/100) * 60
                        row.satel_era5_diff_mins = \
                                satel_minute - grb_minute

                        if satel_name != 'sentinel_1':
                            lat1, lat2, lon1, lon2 = \
                                    self._get_era5_corners_of_rss_cell(
                                        row.lon, row.lat)
                            if lon2 == 360:
                                lon2 = 0
                            corners = []
                            for lat in [lat1, lat2]:
                                for lon in [lon1, lon2]:
                                    lat_idx = self.grid_pts['era5']\
                                            ['lat'].index(lat)
                                    lon_idx = self.grid_pts['era5']\
                                            ['lon'].index(lon)
                                    corners.append(
                                        data[lat_idx][lon_idx])

                            value = float(sum(corners) / len(corners))
                            setattr(row, name, value)
                        else:
                            lat1, lat2, lon1, lon2 = \
                                    self._get_era5_corners_of_scs_cell(
                                        row.lon, row.lat)

                            corners = []
                            for lat in [lat1, lat2]:
                                row_of_lat = []
                                for lon in [lon1, lon2]:
                                    lat_idx = self.grid_pts['era5']\
                                            ['lat'].index(lat)
                                    lon_idx = self.grid_pts['era5']\
                                            ['lon'].index(lon)

                                    row_of_lat.append(
                                        data[lat_idx][lon_idx])
                                corners.append(row_of_lat)

                            f = scipy.interpolate.interp2d(
                                [lon1, lon2], [lat1, lat2], corners)

                            # From x * 0.25 to (x + 1) * 0.25,
                            # there are 6 points at the intervel of
                            # 0.05
                            xnew = [x * 0.05 + lon1 for x in range(6)]
                            ynew = [y * 0.05 + lat1 for y in range(6)]
                            znew = f(xnew, ynew)

                            x_idx = xnew.index(row.lon)
                            y_idx = ynew.index(row.lat)
                            value = float(znew[y_idx][x_idx])
                            setattr(row, name, value)

            grbidx.close()

        utils.delete_last_lines()
        print('Done')

        if satel_name == 'amsr2' or satel_name == 'smap':
            satel_part = self.decompose_radiometer_wind_by_era5(
                satel_name, satel_part)
            if satel_name == 'amsr2':
                satel_part = self.cal_avg_u_v_wind_of_amsr2(satel_part)

        return satel_part

    def cal_avg_u_v_wind_of_amsr2(self, satel_part):
        for row in satel_part:
            row.u_wind_avg = 0.5 * (row.u_wind_lf + row.u_wind_mf)
            row.v_wind_avg = 0.5 * (row.v_wind_lf + row.v_wind_mf)

        return satel_part

    def decompose_radiometer_wind_by_era5(self, satel_name,
                                          satel_part):
        windspd_and_uv_col_name = {
            'amsr2': {
                'wind_lf': ['u_wind_lf', 'v_wind_lf'],
                'wind_mf': ['u_wind_mf', 'v_wind_mf']
            },
            'smap': {
                'windspd': ['u_wind', 'v_wind']
            }
        }

        for row in satel_part:
            winddir = math.degrees(
                math.atan2(row.u_component_of_wind,
                           row.v_component_of_wind)
            )
            for windspd_name in windspd_and_uv_col_name[satel_name]:
                windspd = getattr(row, windspd_name)
                u_wind, v_wind = utils.decompose_wind(windspd, winddir,
                                                      'o')
                setattr(row, windspd_and_uv_col_name[satel_name]\
                        [windspd_name][0], u_wind)
                setattr(row, windspd_and_uv_col_name[satel_name]\
                        [windspd_name][1], v_wind)

        return satel_part

    def _get_era5_corners_of_scs_cell(self, lon, lat):
        if lon < 0 or lat < 0:
            self.logger.error((f"""SCS grid point's lon or """
                               f"""lat is negative"""))

        lon_frac_part, lon_inte_part = math.modf(lon)
        lat_frac_part, lat_inte_part = math.modf(lat)

        era5_frac_parts = np.arange(0.0, 1.01, 0.25)

        for start_idx in range(len(era5_frac_parts) - 1):
            start = era5_frac_parts[start_idx]
            end = era5_frac_parts[start_idx + 1]

            if lon_frac_part >= start and lon_frac_part < end:
                lon1 = start + lon_inte_part
                lon2 = end + lon_inte_part

            if lat_frac_part >= start and lat_frac_part < end:
                lat1 = start + lat_inte_part
                lat2 = end + lat_inte_part

        try:
            return lat1, lat2, lon1, lon2
        except NameError:
            self.logger.error((f"""Fail getting ERA5 corners """
                               f"""of SCS grid cell"""))

    def _get_era5_corners_of_rss_cell(self, lon, lat):
        delta = self.CONFIG['rss']['spatial_resolution'] * 0.5
        lat1 = lat - delta
        lat2 = lat + delta
        lon1 = lon - delta
        lon2 = lon + delta

        return lat1, lat2, lon1, lon2

    def _get_hourtime_row_dict(self, satel_part):
        """Generate a dict to record the relationship between SMAP data
        point and its closest hour.

        """
        satel_day = satel_part[0].satel_datetime.day
        hourtime_row = dict()
        next_day_indices = []

        for hourtime in range(0, 2400, 100):
            hourtime_row[hourtime] = []

        for idx, row in enumerate(satel_part):
            hour_roundered_dt = utils.hour_rounder(row.satel_datetime)
            closest_time = 100 * hour_roundered_dt.hour
            hourtime_row[closest_time].append(idx)

            if hour_roundered_dt.day != satel_day:
                next_day_indices.append(idx)

        return hourtime_row, next_day_indices

    def download(self, satel_name, target_datetime):
        """Download satellite on specified date of TC.

        """
        if not self._datetime_in_satel_lifetime(satel_name,
                                                target_datetime):
            return None

        satel_config = self.CONFIG[satel_name]
        satel_date = target_datetime.date()

        if satel_name != 'sentinel_1':

            utils.setup_signal_handler()
            file_path = self._download_satel_data_in_specified_date(
                satel_config, satel_name, satel_date)
            utils.reset_signal_handler()

        else:

            self._setup_signal_handler()
            file_path = self._download_sentinel_1_in_specified_date(
                satel_config, satel_date, have_uuid_table=True)
            self._reset_signal_handler()

        return file_path

    def _setup_signal_handler(self):
        """Arrange handler for several signal.

        Parameters
        ----------
        None
            Nothing required by this function.

        Returns
        -------
        None
            Nothing returned by this function.

        """
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGHUP, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _reset_signal_handler(self):
        global current_file
        current_file = None

        signal.signal(signal.SIGINT, signal.default_int_handler)

    def _handler(self, signum, frame):
        """Handle forcing quit which may be made by pressing Control + C and
        sending SIGINT which will interupt this application.

        Parameters
        ----------
        signum : int
            Signal number which is sent to this application.
        frame : ?
            Current stack frame.

        Returns
        -------
        None
            Nothing returned by this function.

        """
        # Remove file that is downloaded currently in case forcing quit
        # makes this file uncomplete
        global current_file
        if current_file is not None:
            os.remove(current_file)
            info = f'Removing uncompleted downloaded file: {current_file}'
            self.logger.info(info)
        # Print log
        print('\nForce quit on %s.\n' % signum)
        # Force quit
        sys.exit(1)

    def _get_all_sentinel_1_uuid_and_datetime(self, satel_config):

        for dir in satel_config['dirs']:
            os.makedirs(satel_config['dirs'][dir], exist_ok=True)

        wget_str, query_res_file = self._generate_wget_str(
            satel_config, 'first_try.xml', 0, 10)

        # Call wget to get targets' UUID
        os.system(wget_str)

        total_results, namespace = self.get_query_total_and_namespace(
            query_res_file)

        if not total_results:
            return None

        DatetimeUUID = self.create_datetime_uuid_table()
        datetime_uuid = []

        rows = 100
        for i in range(math.ceil(total_results /
                                 satel_config['query_rows'])):
            start = 100 * i
            self.logger.debug((f"""Downloading and parsing {start} - """
                               f"""{start+99} rows of query result."""))
            wget_str, query_res_file = self._generate_wget_str(
                satel_config, f'query_result_{start}_{start+99}.xml',
                start, rows)

            os.system(wget_str)

            tree = ET.parse(query_res_file)
            root = tree.getroot()
            for entry in root.findall(f'./{namespace}entry'):

                row = DatetimeUUID()

                row.filename = entry.getchildren()[0].text
                row.uuid = entry.getchildren()[-1].text

                for date_ in entry.findall(f'./{namespace}date'):
                    if date_.get('name') == 'beginposition':
                        row.beginposition = self.date_parser(date_.text)
                    elif date_.get('name') == 'endposition':
                        row.endposition = self.date_parser(date_.text)

                datetime_uuid.append(row)

        utils.bulk_insert_avoid_duplicate_unique(
            datetime_uuid,
            int(self.CONFIG['database']['batch_size']['insert']/10),
            DatetimeUUID, ['uuid'], self.session,
            check_self=True)

    def create_datetime_uuid_table(self):
        table_name = 'scs_sentinel_1_datetime_uuid'

        class DatetimeUUID(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(DatetimeUUID, t)

            return DatetimeUUID

        cols = []
        # IBTrACS columns
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('beginposition', DateTime, nullable=False))
        cols.append(Column('endposition', DateTime, nullable=False))
        cols.append(Column('uuid', String(50), nullable=False,
                           unique=True))
        cols.append(Column('filename', String(80), nullable=False,
                           unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(DatetimeUUID, t)

        self.session.commit()

        return DatetimeUUID

    def date_parser(self, date_str):
        # 2014-12-30T13:31:51.933Z
        dt = datetime.datetime.strptime(
            date_str.split('.')[0].split('Z')[0], '%Y-%m-%dT%H:%M:%S')
        return dt

    def _download_sentinel_1_in_specified_date(self, satel_config,
                                               satel_date, have_uuid_table):
        data_paths = []

        if not have_uuid_table:
            self._get_all_sentinel_1_uuid_and_datetime(satel_config)

        # Traverse all UUIDs to download all target files
        DatetimeUUID = self.create_datetime_uuid_table()

        cwd = os.getcwd()
        os.chdir(satel_config['dirs']['ncs'])

        the_day = satel_date
        for row in self.session.query(DatetimeUUID).filter(
            extract('year', DatetimeUUID.beginposition) == the_day.year,
            extract('month', DatetimeUUID.beginposition) == the_day.month,
            extract('day', DatetimeUUID.beginposition) == the_day.day
        ).yield_per(self.CONFIG['database']['batch_size']['query']):
            #
            file_path = self.download_sentinel_1(satel_config, row.uuid,
                                                 row.filename)
            if file_path is not None:
                data_paths.append(file_path)

        os.chdir(cwd)

        if len(data_paths):
            return data_paths
        else:
            return None

    def _generate_wget_str(self, satel_config, query_result_file_name,
                           start, rows):
        query_res_file = (f"""{satel_config['dirs']['query_results']}"""
                          f"""{query_result_file_name}""")

        # Generate query url
        query_parameters = (f"""(platformname:Sentinel-1 """
                            f"""AND footprint:\\"Intersects(POLYGON(("""
                            f"""{satel_config['aoi']})))\\" """
                            f"""AND producttype:OCN)"""
                            f"""&orderby=beginposition asc""")

        wget_str = (f"""wget --no-check-certificate """
                    f"""--user='{satel_config['user']}' """
                    f"""--password='{satel_config['password']}' """
                    f"""--output-document='{query_res_file}' """
                    f"""\"{satel_config['urls']['query']}"""
                    f"""start={start}"""
                    f"""&rows={rows}"""
                    f"""&q={query_parameters}\" """)

        return wget_str, query_res_file

    def download_sentinel_1(self, satel_config, uuid, filename):

        cwd = os.getcwd()
        test_existence_file_path = (f"""{cwd}/{filename}.zip""")

        if os.path.exists(test_existence_file_path):
            return test_existence_file_path

        file_path = None
        # Checking products availability with OData
        url = (f"""https://scihub.copernicus.eu/dhus/odata/v1/"""
               f"""Products('{uuid}')/Online/$value""")
        page = requests.get(url, auth=(satel_config['user'],
                                       satel_config['password']))
        if page.status_code != 200:
            self.logger.error((f"""Failed checking Sentinel products """
                               f"""availability with OData: """
                               f"""{url}"""))
            return file_path

        online_str = page.text
        need_trigger_by_wget = False

        if online_str == 'false':
            self.logger.info((f"""Sentinel product is offline: """
                              f"""{filename}"""))

            # Triggering the retrieval of offline products
            trigger_url = (f"""https://scihub.copernicus.eu/dhus/odata/"""
                           f"""v1/Products('{uuid}')/$value""")
            trigger_page = requests.get(
                url, auth=(satel_config['user'],
                           satel_config['password']))

            if trigger_page.status_code == 202:
                self.logger.info((f"""Successfully triggers the """
                                  f"""retrieval of offline product: """
                                  f"""{filename}"""))
            else:
                self.logger.debug((f"""Failed SIMPLY triggering the """
                                  f"""retrieval of offline product: """
                                  f"""{filename}"""))
                need_trigger_by_wget = True

        if online_str == 'true' or need_trigger_by_wget:
            if online_str == 'true':
                file_path = test_existence_file_path
                self.logger.debug((f"""Downloading Sentinel product: """
                                   f"""{filename}"""))

            elif need_trigger_by_wget:
                self.logger.info((f"""Using wget to trigger the """
                                  f"""retrieval of offline product: """
                                  f"""{filename}"""))

            wget_str = (f"""wget --content-disposition --continue """
                        f"""--user='{satel_config['user']}' """
                        f"""--password='{satel_config['password']}' """
                        f"""\"{satel_config['urls']['download']['prefix']}"""
                        f"""{uuid}"""
                        f"""{satel_config['urls']['download']['suffix']}\" """)
            os.system(wget_str)

        return file_path

    def get_query_total_and_namespace(self, xmlfile):
        total = 0
        # create element tree object
        tree = ET.parse(xmlfile)
        # get root element
        root = tree.getroot()
        for child in root.getchildren():
            if child.tag.endswith('subtitle'):
                total_search = re.search('of (.*) total', child.text)
                total = int(total_search.group(1))
                break

        namespace = re.match(r'\{.*\}', root.tag).group(0)

        return total, namespace

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
        if satel_name != 'smap':
            save_dir = config['dirs']['bmaps']
        else:
            save_dir = config['dirs']['ncs']
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

    def create_ascat_era5_table(self, dt):
        table_name = utils.gen_satel_era5_tablename('ascat', dt)

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return Satel

        cols = utils.get_basic_satel_era5_columns()

        cols.append(Column('windspd', Float, nullable=False))
        cols.append(Column('winddir', Float, nullable=False))

        cols.append(Column('u_wind', Float, nullable=False))
        cols.append(Column('v_wind', Float, nullable=False))

        cols.append(Column('scatflag', Boolean, nullable=False))
        cols.append(Column('radrain', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, work=False,
                                 save_disk=self.save_disk)
        era5_cols = era5_.get_era5_columns()
        cols = cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(Satel, t)

        self.session.commit()

        return Satel

    def create_wsat_era5_table(self, dt):
        table_name = utils.gen_satel_era5_tablename('wsat', dt)

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return Satel

        cols = utils.get_basic_satel_era5_columns()

        cols.append(Column('w_lf', Float, nullable=False))
        cols.append(Column('w_mf', Float, nullable=False))
        cols.append(Column('w_aw', Float, nullable=False))
        cols.append(Column('winddir', Float, nullable=False))

        cols.append(Column('u_wind_lf', Float, nullable=False))
        cols.append(Column('v_wind_lf', Float, nullable=False))
        cols.append(Column('u_wind_mf', Float, nullable=False))
        cols.append(Column('v_wind_mf', Float, nullable=False))
        cols.append(Column('u_wind_aw', Float, nullable=False))
        cols.append(Column('v_wind_aw', Float, nullable=False))

        cols.append(Column('sst', Float, nullable=False))
        cols.append(Column('vapor', Float, nullable=False))
        cols.append(Column('cloud', Float, nullable=False))
        cols.append(Column('rain', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, work=False,
                                 save_disk=self.save_disk)
        era5_cols = era5_.get_era5_columns()
        cols = cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(Satel, t)

        self.session.commit()

        return Satel

    def create_amsr2_era5_table(self, dt):
        table_name = utils.gen_satel_era5_tablename('amsr2', dt)

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return Satel

        cols = utils.get_basic_satel_era5_columns()

        cols.append(Column('wind_lf', Float, nullable=False))
        cols.append(Column('wind_mf', Float, nullable=False))

        cols.append(Column('u_wind_lf', Float, nullable=False))
        cols.append(Column('v_wind_lf', Float, nullable=False))
        cols.append(Column('u_wind_mf', Float, nullable=False))
        cols.append(Column('v_wind_mf', Float, nullable=False))
        cols.append(Column('u_wind_avg', Float, nullable=False))
        cols.append(Column('v_wind_avg', Float, nullable=False))

        cols.append(Column('sst', Float, nullable=False))
        cols.append(Column('vapor', Float, nullable=False))
        cols.append(Column('cloud', Float, nullable=False))
        cols.append(Column('rain', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, work=False,
                                 save_disk=self.save_disk)
        era5_cols = era5_.get_era5_columns()
        cols = cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(Satel, t)

        self.session.commit()

        return Satel

    def create_smap_era5_table(self, dt):
        table_name = utils.gen_satel_era5_tablename('smap', dt)

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return Satel

        cols = utils.get_basic_satel_era5_columns()

        cols.append(Column('windspd', Float, nullable=False))

        cols.append(Column('u_wind', Float, nullable=False))
        cols.append(Column('v_wind', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, work=False,
                                 save_disk=self.save_disk)
        era5_cols = era5_.get_era5_columns()
        cols = cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(Satel, t)

        self.session.commit()

        return Satel

    def create_sentinel_1_table(self, dt):
        table_name = utils.gen_satel_era5_tablename('sentinel_1', dt)

        class SatelERA5(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(SatelERA5, t)

            return SatelERA5

        cols = utils.get_basic_satel_era5_columns()


        cols.append(Column('ecmwf_windspd', Float, nullable=False))
        cols.append(Column('ecmwf_winddir', Float, nullable=False))
        cols.append(Column('ecmwf_u_wind', Float, nullable=False))
        cols.append(Column('ecmwf_v_wind', Float, nullable=False))

        cols.append(Column('windspd', Float, nullable=False))
        cols.append(Column('winddir', Float, nullable=False))
        cols.append(Column('u_wind', Float, nullable=False))
        cols.append(Column('v_wind', Float, nullable=False))

        cols.append(Column('inversion_quality', Float, nullable=False))
        cols.append(Column('wind_quality', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, work=False,
                                 save_disk=self.save_disk)
        era5_cols = era5_.get_era5_columns()
        cols = cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(SatelERA5, t)

        self.session.commit()

        return SatelERA5

    def get_latlon_and_match_index(self, lat_or_lon, latlon_idx):
        if lat_or_lon == 'lat':
            lat_of_row = self.grid_pts['rss']['lat']\
                    [self.lat1_index + latlon_idx]
            # Choose south grid point nearest to RSS point
            lat_match_index = self.grid_lats.index(
                    lat_of_row - 0.5 * self.spa_resolu['grid'])

            return lat_of_row, lat_match_index

        elif lat_or_lon == 'lon':
            lon_of_pt = self.grid_pts['rss']['lon']\
                    [self.lon1_index + latlon_idx]
            # Choose east grid point nearest to RSS point
            lon_match_index = self.grid_lons.index(
                    lon_of_pt + 0.5 * self.spa_resolu['grid'])

            return lon_of_pt, lon_match_index

    def _get_ascat_part(self, vars, SatelERA5, target_date, missing):
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
                lat_of_row, lat_match_index = \
                        self.get_latlon_and_match_index('lat', y)

                for x in range(lons_num):
                    # Skip when the cell has no data
                    if subset['nodata'][i][y][x]:
                        continue

                    # Skip when the time of ascending and descending
                    # passes is same
                    if subset['mingmt'][0][y][x] == \
                       subset['mingmt'][1][y][x]:
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

                    row = SatelERA5()
                    row.satel_datetime = datetime.datetime.combine(
                        target_date, time_)

                    lon_of_pt, lon_match_index = \
                            self.get_latlon_and_match_index('lon', x)

                    row.x = int(self.grid_x[lon_match_index])
                    row.y = int(self.grid_y[lat_match_index])

                    row.lon = lon_of_pt
                    row.lat = lat_of_row

                    row.satel_datetime_lon_lat = (
                        f'{row.satel_datetime}_{row.lon}_{row.lat}')
                    row.scatflag = (
                        None if subset['scatflag'][i][y][x] == missing
                        else bool(subset['scatflag'][i][y][x])
                    )
                    row.radrain = (
                        None if subset['radrain'][i][y][x] < 0
                        else float(subset['radrain'][i][y][x])
                    )
                    row.windspd = (
                        None if subset['windspd'][i][y][x] == missing
                        else float(subset['windspd'][i][y][x])
                    )
                    row.winddir = (
                        None if subset['winddir'][i][y][x] == missing
                        else float(subset['winddir'][i][y][x])
                    )
                    row.u_wind, row.v_wind = utils.decompose_wind(
                        row.windspd, row.winddir, 'o')

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

    def _get_wsat_part(self, vars, SatelERA5, target_date, missing):
        satel_data = []

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['nodata', 'mingmt', 'sst', 'vapor', 'cloud', 'rain',
                     'land', 'ice', 'w-lf', 'w-mf', 'w-aw', 'wdir']

        for var_name in var_names:
            subset[var_name] = vars[var_name][
                :,
                self.lat1_index: self.lat2_index+1,
                self.lon1_index: self.lon2_index+1]

        passes_num, lats_num, lons_num = subset[var_names[0]].shape

        for i in range(passes_num):
            for y in range(lats_num):
                lat_of_row, lat_match_index = \
                        self.get_latlon_and_match_index('lat', y)

                for x in range(lons_num):
                    # Skip when the cell has no data
                    if subset['nodata'][i][y][x]:
                        continue

                    # Skip when the time of ascending and descending passes
                    # is same and (anyone of wind speed variabkles 
                    # or wdir is not the same)
                    if subset['mingmt'][0][y][x] == \
                       subset['mingmt'][1][y][x]:
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
                    time_ = datetime.time(
                        *divmod(int(subset['mingmt'][i][y][x]),60), 0)

                    row = SatelERA5()
                    row.satel_datetime = datetime.datetime.combine(
                        target_date, time_)

                    lon_of_pt, lon_match_index = \
                            self.get_latlon_and_match_index('lon', x)

                    row.x = int(self.grid_x[lon_match_index])
                    row.y = int(self.grid_y[lat_match_index])

                    row.lon = lon_of_pt
                    row.lat = lat_of_row

                    row.satel_datetime_lon_lat = (f'{row.satel_datetime}'
                                                  + f'_{row.lon}'
                                                  + f'_{row.lat}')

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

                    row.u_wind_lf, row.v_wind_lf = utils.decompose_wind(
                        row.w_lf, row.winddir, 'o')
                    row.u_wind_mf, row.v_wind_mf = utils.decompose_wind(
                        row.w_mf, row.winddir, 'o')
                    row.u_wind_aw, row.v_wind_aw = utils.decompose_wind(
                        row.w_aw, row.winddir, 'o')

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

    def _get_amsr2_part(self, vars, SatelERA5, target_date, missing):
        satel_data = []

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['nodata', 'time', 'sst', 'vapor', 'cloud', 'rain',
                     'land', 'ice', 'windLF', 'windMF']

        for var_name in var_names:
            subset[var_name] = vars[var_name][
                :,
                self.lat1_index: self.lat2_index+1,
                self.lon1_index: self.lon2_index+1]

        passes_num, lats_num, lons_num = subset[var_names[0]].shape

        for i in range(passes_num):
            for y in range(lats_num):

                lat_of_row, lat_match_index = \
                        self.get_latlon_and_match_index('lat', y)

                for x in range(lons_num):
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
                    time_ = datetime.time(
                        *divmod(int(60*subset['time'][i][y][x]),60), 0)

                    row = SatelERA5()
                    row.satel_datetime = datetime.datetime.combine(
                        target_date, time_)

                    lon_of_pt, lon_match_index = \
                            self.get_latlon_and_match_index('lon', x)

                    row.x = int(self.grid_x[lon_match_index])
                    row.y = int(self.grid_y[lat_match_index])

                    row.lon = lon_of_pt
                    row.lat = lat_of_row

                    row.satel_datetime_lon_lat = (f'{row.satel_datetime}'
                                                  + f'_{row.lon}'
                                                  + f'_{row.lat}')

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
                    row.wind_lf = (None
                                  if subset['windLF'][i][y][x] == missing
                                  else float(subset['windLF'][i][y][x]))
                    row.wind_mf = (None
                                  if subset['windMF'][i][y][x] == missing
                                  else float(subset['windMF'][i][y][x]))
                    # Wait to be updated when adding ERA5 data
                    row.u_wind_lf = 0
                    row.v_wind_lf = 0
                    row.u_wind_mf = 0
                    row.v_wind_mf = 0
                    row.u_wind_avg = 0
                    row.v_wind_avg = 0

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

    def _get_smap_part(self, vars, SatelERA5, target_date, missing):
        satel_data = []

        # In bytemaps, vars['latitude'][0] is the sourthest latitude
        # vars['longitude'][0] is the minimal positive longitude
        subset = dict()
        var_names = ['minute', 'wind']

        for var_name in var_names:
            subset[var_name] = vars[var_name][
                self.lat1_index: self.lat2_index+1,
                self.lon1_index: self.lon2_index+1,
                :
            ]

        lats_num, lons_num, passes_num = subset[var_names[0]].shape

        for y in range(lats_num):

            lat_of_row, lat_match_index = \
                    self.get_latlon_and_match_index('lat', y)

            for x in range(lons_num):
                for i in range(passes_num):
                    # Skip when the cell has no data
                    if (subset['minute'][y][x][i] == missing['minute']
                        or subset['wind'][y][x][i] == missing['wind']):
                        continue

                    # Skip when the time of ascending and descending passes
                    # is same and (w-aw or wdir is not the same)
                    if subset['minute'][y][x][0] == vars['minute'][y][x][1]:
                        continue

                    # Process the datetime and skip if necessary
                    time_ = datetime.time(
                        *divmod(int(subset['minute'][y][x][i]), 60), 0)

                    row = SatelERA5()
                    row.satel_datetime = datetime.datetime.combine(
                        target_date, time_)

                    lon_of_pt, lon_match_index = \
                            self.get_latlon_and_match_index('lon', x)

                    row.x = int(self.grid_x[lon_match_index])
                    row.y = int(self.grid_y[lat_match_index])

                    row.lon = lon_of_pt
                    row.lat = lat_of_row

                    row.satel_datetime_lon_lat = (f'{row.satel_datetime}'
                                                  + f'_{row.lon}'
                                                  + f'_{row.lat}')

                    row.windspd = float(subset['wind'][y][x][i])
                    # Wait to be updated when adding ERA5 data
                    row.u_wind = 0
                    row.v_wind = 0

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

    def _get_sentinel_1_part(self, zip_file_paths, SatelERA5,
                             target_date):
        data_dir = os.path.dirname(zip_file_paths[0])
        all_data = []

        for zip_file in zip_file_paths:
            if not zip_file.endswith('83C3.zip'):
                continue

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

            nc_files_dir = (f"""{zip_file.replace('zip', 'SAFE')}/"""
                            f"""measurement/""")
            nc_file_names = [f for f in os.listdir(nc_files_dir)
                             if f.endswith('.nc')]

            for nc_name in nc_file_names:
                nc_path = f'{nc_files_dir}{nc_name}'

                data = self.read_single_sentinel_1_nc(nc_path, SatelERA5)

                all_data += data

        return all_data

    def read_single_sentinel_1_nc(self, nc_path, SatelERA5):
        self.logger.debug((f"""Reading Sentinel-1 data from """
                           f"""{nc_path}"""))
        data = []

        nc_name_parts = nc_path.split('/')[-1].split('-')
        start, end = [x for x in nc_name_parts if len(x) == 15]

        start_dt = datetime.datetime.strptime(start, '%Y%m%dt%H%M%S')
        end_dt = datetime.datetime.strptime(end, '%Y%m%dt%H%M%S')

        mean_dt = start_dt + (end_dt - start_dt) / 2
        mean_dt = mean_dt.replace(microsecond = 0)

        dataset = netCDF4.Dataset(nc_path)
        vars = dataset.variables

        owi_az_size, owi_ra_size = vars['owiLat'].shape

        lons, lats = vars['owiLon'][:], vars['owiLat'][:]

        try:
            # Aiming at South China Sea
            # So lon is positive and lat is not negative
            min_lon = float(lons[lons > 0].min())
            max_lon = float(lons[lons > 0].max())
            min_lat = float(lats[lats >= 0].min())
            max_lat = float(lats[lats >= 0].max())
        except ValueError:
            # If there is no positive lon or no nonnegative lat,
            # sentinel-1 data area is out of SCS
            self.logger.debug((f"""Data area is out of target """
                               f"""region: {nc_path}"""))
            return data

        self.ocean_grid_pts = self.get_grid_pts_in_range(
            min_lon, max_lon, min_lat, max_lat)
        grid_pts_num = self.ocean_grid_pts.count()

        self.pt_region_data_indices = []

        subprocess_num = 4
        subset_pts_num = int(grid_pts_num / subprocess_num)

        parallelism_workload = []
        for i in range(subprocess_num):
            start = i * subset_pts_num
            if i != subprocess_num - 1:
                end = (i + 1) * subset_pts_num
            else:
                end = grid_pts_num

            workload = utils.GetChildDataIndices(
                self.ocean_grid_pts[start:end], lats, lons,
                owi_az_size)
            parallelism_workload.append(workload)

        with Pool(subprocess_num) as p:
            parallelism_res = p.map(
                utils.get_data_indices_around_grid_pts,
                parallelism_workload
            )

        for res_subset in parallelism_res:
            for pt_res in res_subset:
                self.pt_region_data_indices.append(pt_res)

        print(f'ocean gird pts num: {grid_pts_num}')
        print((f"""len of pt_region_data_indices: """
               f"""{len(self.pt_region_data_indices)}"""))

        var_names = [
            'owiEcmwfWindSpeed', 'owiEcmwfWindDirection',
            'owiWindSpeed', 'owiWindDirection',
            'owiInversionQuality', 'owiWindQuality'
        ]

        total = grid_pts_num
        for pt_idx, pt in enumerate(self.ocean_grid_pts):
            percent = float(pt_idx + 1) / total * 100
            print(f'\r{percent:.2f}%', end='')

            ecmwf_windspd = []
            ecmwf_winddir = []
            windspd = []
            winddir = []
            # :flag_values = 0B, 1B, 2B; // byte
            # :flag_meanings = "good medium poor"
            inversion_quality = []
            # :flag_values = 0B, 1B, 2B, 3B; // byte
            # :flag_meanings = "good medium low poor"
            wind_quality = []

            for index_pair in self.pt_region_data_indices[pt_idx]:
                i, j = index_pair[0], index_pair[1]
                # Mask
                if vars['owiMask'][i][j] != 0.0:
                    continue

                skip = False
                for var_name in var_names:
                    if vars[var_name][i][j] is MASKED:
                        skip = True
                        break
                if skip:
                    continue

                ecmwf_windspd.append(vars['owiEcmwfWindSpeed'][i][j])
                ecmwf_winddir.append(vars['owiEcmwfWindDirection'][i][j])
                windspd.append(vars['owiWindSpeed'][i][j])
                winddir.append(vars['owiWindDirection'][i][j])
                inversion_quality.append(
                    float(vars['owiInversionQuality'][i][j]))
                wind_quality.append(
                    float(vars['owiWindQuality'][i][j]))

            if not len(windspd) or not len(winddir):
                continue

            row = SatelERA5()
            row.satel_datetime = mean_dt
            row.x = pt.x
            row.y = pt.y
            row.lon = pt.lon
            row.lat = pt.lat
            row.satel_datetime_lon_lat = (f"""{row.satel_datetime}"""
                                          + f"""_{row.lon}_{row.lat}""")
            row.ecmwf_windspd = float(mean(ecmwf_windspd))
            # In Sentinel-1's NetCDF file, owiEcmwfWindDirection is
            # meteorological convention, which needed to be converted
            # to oceangraphic convention
            row.ecmwf_winddir = (float(mean(ecmwf_winddir)) + 180) % 360
            row.ecmwf_u_wind, row.ecmwf_v_wind = utils.decompose_wind(
                row.ecmwf_windspd, row.ecmwf_winddir, 'o')

            row.windspd = float(mean(windspd))
            # In Sentinel-1's NetCDF file, owiWindDirection is
            # meteorological convention, which needed to be converted
            # to oceangraphic convention
            row.winddir = (float(mean(winddir)) + 180) % 360
            row.u_wind, row.v_wind = utils.decompose_wind(
                row.windspd, row.winddir, 'o')

            row.inversion_quality = mean(inversion_quality)
            row.wind_quality = mean(wind_quality)

            data.append(row)

        utils.delete_last_lines()
        print('Done')

        return data
