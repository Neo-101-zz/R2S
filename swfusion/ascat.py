"""Manage downloading and reading ASCAT, QucikSCAT and Windsat data.

"""
import datetime
import math
import logging
import pickle
import operator
import re
import os
import time
import sys
import xml.etree.ElementTree as ET

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
        # self.satel_names = ['ascat', 'wsat', 'amsr2', 'smap']
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

    def create_satel_era5_table(self, satel_name, target_datetime):
        if satel_name == 'ascat':
            SatelERA5 = self.create_ascat_era5_table(target_datetime)
        elif satel_name == 'wsat':
            SatelERA5 = self.create_wsat_era5_table(target_datetime)
        elif satel_name == 'amsr2':
            SatelERA5 = self.create_amsr2_era5_table(target_datetime)
        elif satel_name == 'smap':
            SatelERA5 = self.create_smap_era5_table(target_datetime)

        return SatelERA5

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
                satel_data_path = self.download(satel_name, target_datetime)
                utils.reset_signal_handler()
                if satel_data_path is None:
                    continue
                # Download ERA5 data
                self.logger.debug((f'Downloading all surface ERA5 '
                                   + f'data of all hours on '
                                   + f'{target_datetime.date()}'))

                # Get satellite table
                SatelERA5 = self.create_satel_era5_table(satel_name,
                                                         target_datetime)
                self.logger.debug((f'Reading {satel_name} and ERA5 '
                                   + f'data on '
                                   + f'{target_datetime.date()}'))
                self.read_satel_era5(satel_name, satel_data_path,
                                     era5_data_path, SatelERA5,
                                     target_datetime.date())
            if self.save_disk:
                os.remove(era5_data_path['today'])
            breakpoint()

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
        satel_era5_data = self._add_era5_part(satel_part, era5_data_path,
                                              target_date)

        return satel_era5_data

    def _get_satel_part(self, satel_name, satel_data_path, SatelERA5,
                        target_date):
        self.logger.info((f'Getting {satel_name} part from '
                         + f'{satel_data_path}'))
        missing = self.CONFIG[satel_name]['missing_value']

        if satel_name != 'smap':
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
        else:
            dataset = netCDF4.Dataset(satel_data_path)
            # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which faster
            # than 1 m/s, so must disable auto mask
            dataset.set_auto_mask(False)
            vars = dataset.variables

            satel_part = self._get_smap_part(vars, SatelERA5,
                                              target_date, missing)

        return satel_part

    def _add_era5_part(self, satel_part, era5_data_path, target_date):
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

            grbidx.close()

        utils.delete_last_lines()
        print('Done')

        return satel_part

    def _get_corners_of_cell(self, lon, lat):
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
            file_path = self._download_satel_data_in_specified_date(
                satel_config, satel_name, satel_date)
        else:
            file_path = self._download_sentinel_1_in_specified_date(
                satel_config, satel_date)

        return file_path

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
            wget_str, query_res_file = self._generate_wget_str(
                satel_config, f'query_result_{start}_{start+99}.xml',
                start, rows)

            os.system(wget_str)

            tree = ET.parse(query_res_file)
            root = tree.getroot()
            for entry in root.findall(f'./{namespace}entry'):

                row = DatetimeUUID()

                row.uuid = entry.getchildren()[-1].text

                for date_ in entry.findall(f'./{namespace}date'):

                    if date_.get('name') == 'beginposition':
                        row.beginposition = self.date_parser(date_.text)
                    elif date_.get('name') == 'endposition':
                        row.endposition = self.date_parser(date_.text)
                datetime_uuid.append(row)

        self.datetime_uuid = datetime_uuid

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

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(DatetimeUUID, t)

        self.session.commit()

        return DatetimeUUID

    def date_parser(self, date_str):
        # 2014-12-30T13:31:51.933Z
        dt = datetime.datetime.strptime(date_str.split('.')[0],
                                        '%Y-%m-%dT%H:%M:%S')
        return dt

    def _download_sentinel_1_in_specified_date(self, satel_config,
                                               satel_date):
        data_paths = []

        self._get_all_sentinel_1_uuid_and_datetime(satel_config)

        # Traverse all UUIDs to download all target files

        return data_paths

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
        table_name = f'ascat_{dt.year}_{str(dt.month).zfill(2)}'

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
        table_name = f'wsat_{dt.year}_{str(dt.month).zfill(2)}'

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
        table_name = f'amsr2_{dt.year}_{str(dt.month).zfill(2)}'

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return Satel

        cols = utils.get_basic_satel_era5_columns()
        cols.append(Column('windLF', Float, nullable=False))
        cols.append(Column('windMF', Float, nullable=False))
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
        table_name = f'smap_{dt.year}_{str(dt.month).zfill(2)}'

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return Satel

        cols = utils.get_basic_satel_era5_columns()
        cols.append(Column('windspd', Float, nullable=False))

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

                    row = SatelERA5()
                    row.satel_datetime = datetime.datetime.combine(
                        target_date, time_)
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[self.lon1_index + x]
                    row.lat = self.lat_grid_points[self.lat1_index + y]
                    row.satel_datetime_lon_lat = (f'{row.satel_datetime}'
                                                  + f'_{row.lon}'
                                                  + f'_{row.lat}')
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
                for x in range(lons_num):
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
                    time_ = datetime.time(
                        *divmod(int(subset['mingmt'][i][y][x]),60), 0)

                    row = SatelERA5()
                    row.satel_datetime = datetime.datetime.combine(
                        target_date, time_)
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[self.lon1_index + x]
                    row.lat = self.lat_grid_points[self.lat1_index + y]
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
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[self.lon1_index + x]
                    row.lat = self.lat_grid_points[self.lat1_index + y]
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
                    row.x = x
                    row.y = y
                    row.lon = self.lon_grid_points[self.lon1_index + x]
                    row.lat = self.lat_grid_points[self.lat1_index + y]
                    row.satel_datetime_lon_lat = (f'{row.satel_datetime}'
                                                  + f'_{row.lon}'
                                                  + f'_{row.lat}')

                    row.windspd = float(subset['wind'][y][x][i])

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
