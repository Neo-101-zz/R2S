"""Manage downloading and reading ASCAT, QucikSCAT and Windsat data.

"""
import datetime
import math
import logging
import pickle
import os
import time
import sys

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker

import mysql.connector
from ascat_daily import ASCATDaily
from quikscat_daily_v4 import QuikScatDaily
from windsat_daily_v7 import WindSatDaily
import utils

Base = declarative_base()

class SatelManager(object):

    def __init__(self, CONFIG, period, region, passwd):
        self.logger = logging.getLogger(__name__)
        self.satel_names = ['ascat', 'qscat', 'wsat']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd

        self.download()
        self.read(read_all=True)

    def download(self):
        utils.setup_signal_handler()
        # Check and update period for each satellite
        self.periods = dict()
        self.downloaded_file_path = dict()
        for satel_name in self.satel_names:
            correct, period = utils.check_and_update_period(
                self.period, self.CONFIG[satel_name]['period_limit'],
                self.CONFIG['workflow']['prompt'])
            if not correct:
                return
            self.periods[satel_name] = period
            config4one = self.CONFIG[satel_name]
            self.downloaded_file_path[satel_name] = []
            self._download_single_satel(config4one, satel_name, period)

    def read(self, read_all):
        """Read data into MySQL database.

        """
        utils.reset_signal_handler()

        DB_CONFIG = self.CONFIG['database']
        PROMPT = self.CONFIG['workflow']['prompt']
        DBAPI = DB_CONFIG['db_api']
        USER = DB_CONFIG['user']
        # password_ = input(PROMPT['input']['db_root_password'])
        password_ = self.db_root_passwd
        HOST = DB_CONFIG['host']
        DB_NAME = DB_CONFIG['db_name']
        ARGS = DB_CONFIG['args']

        self.cnx = mysql.connector.connect(user=USER, password=password_,
                                           host=HOST, use_pure=True)
        utils.create_database(self.cnx, DB_NAME)
        utils.use_database(self.cnx, DB_NAME)

        # Define the MySQL engine using MySQL Connector/Python
        connect_string = ('{0}://{1}:{2}@{3}/{4}?{5}'.format(
            DBAPI, USER, password_, HOST, DB_NAME, ARGS))
        self.engine = create_engine(connect_string, echo=False)
        # Create table of cwind station
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        self._insert_satel(read_all)

    def _insert_satel(self, read_all):
        if read_all:
            files_path = dict()
            for satel_name in self.satel_names:
                files_path[satel_name] = []
                files_path[satel_name] += [
                    (self.CONFIG[satel_name]['dirs']['bmaps']
                     + x) for x in os.listdir(
                         self.CONFIG[satel_name]['dirs']['bmaps'])
                    if x.endswith('.gz')
                ]
        else:
            files_path = self.downloaded_file_path

        skip_vars = ['mingmt', 'nodata']
        not_null_vars = ['latitude', 'longitude']
        unique_vars = []
        custom_cols = {1: Column('datetime', DateTime(),
                                 nullable=False, unique=False),
                       -1: Column('space_time', String(255),
                                 nullable=False, unique=True)}

        old_files_path = files_path
        files_path = dict()
        for satel_name in old_files_path.keys():
            files_path[satel_name] = []
            for file_path in old_files_path[satel_name]:
                date_= datetime.datetime.strptime(
                    file_path.split('/')[-1].split('_')[1][:8] + '000000',
                    '%Y%m%d%H%M%S').date()
                if utils.check_period(date_, self.period):
                    files_path[satel_name].append(file_path)

        for satel_name in files_path.keys():
            self.logger.info(self.CONFIG[satel_name]\
                             ['prompt']['info']['read'])
            # Create table of particular satellite
            bytemap_file = files_path[satel_name][0]
            if satel_name == 'ascat' or satel_name == 'qscat':
                not_null_vars += ['windspd', 'winddir']
            elif satel_name == 'wsat':
                not_null_vars += ['w-aw', 'wdir'] 

            total = len(files_path[satel_name])
            count = 0

            for file_path in files_path[satel_name]:
                count += 1
                year_str = file_path.split('/')[-1].split('_')[1][:4]
                table_name = '{0}_{1}'.format(satel_name, year_str)

                SatelTable = utils.create_table_from_bytemap(
                    self.engine, satel_name, bytemap_file,
                    table_name, self.session, skip_vars, not_null_vars,
                    unique_vars, custom_cols)

                info = (f'Extracting {satel_name} data '
                        + f'from {file_path.split("/")[-1]}')
                if count > 1:
                    utils.delete_last_lines()
                print(f'\r{info} ({count}/{total})', end='')

                start = time.process_time()
                one_day_records = self._extract_satel_bytemap(satel_name,
                                                              file_path,
                                                              SatelTable)
                end = time.process_time()
                self.logger.debug(f'{info} in {end-start:.2f} s')

                total_sample = one_day_records
                batch_size = self.CONFIG['database']['batch_size']
                table_class = SatelTable
                unique_cols = ['space_time']
                session = self.session

                start = time.process_time()
                utils.bulk_insert_avoid_duplicate_unique(
                    total_sample, batch_size, table_class, unique_cols,
                    session)
                end = time.process_time()
                self.logger.debug((f'Bulk inserting {satel_name} data '
                                   + f'into {table_name} '
                                   + f'in {end-start:.2f} s'))

            utils.delete_last_lines()
            print(f'Done')

    def _extract_satel_bytemap(self, satel_name, file_path, SatelTable):
        bm_file = file_path
        table_class = SatelTable
        skip_vars = ['mingmt', 'nodata']
        datetime_func = datetime_from_bytemap
        datetime_col_name = 'datetime'
        missing = self.CONFIG[satel_name]['missing_value']
        valid_func = valid_bytemap
        unique_func = utils.gen_space_time_fingerprint
        unique_col_name = 'space_time'
        lat_name = 'latitude'
        lon_name = 'longitude'
        period = self.period
        region = self.region
        not_null_vars = ['latitude', 'longitude']
        if satel_name == 'ascat' or satel_name == 'qscat':
            not_null_vars += ['windspd', 'winddir']
        elif satel_name == 'wsat':
            not_null_vars += ['w_aw', 'wdir']

        # Not recommend to use utils.extract_bytemap_to_table, because it's
        # very slow due to too much fucntion call
        res = self._extract_bytemap_to_table_2(satel_name, bm_file,
                                               table_class, missing)

        return res

    def _extract_bytemap_to_table_2(self, satel_name, bm_file, table_class,
                                    missing):
        dataset = utils.dataset_of_daily_satel(satel_name, bm_file)
        vars = dataset.variables

        min_lat, max_lat = self.region[0], self.region[1]
        min_lon, max_lon = self.region[2], self.region[3]
        min_lat_idx, max_lat_idx = utils.find_index([min_lat, max_lat], 'lat')
        lat_indices = [x for x in range(min_lat_idx, max_lat_idx+1)]
        min_lon_idx, max_lon_idx = utils.find_index([min_lon, max_lon], 'lon')
        lon_indices = [x for x in range(min_lon_idx, max_lon_idx+1)]

        lat_len = len(lat_indices)
        lon_len = len(lon_indices)
        total = 2 * lat_len * lon_len
        count = 0

        # Store all rows
        whole_table = []

        st = time.time()
        iasc = [0, 1]
        # iasc = 0 (morning, descending passes)
        # iasc = 1 (evening, ascending passes)
        for i in iasc:
            for j in lat_indices:
                for k in lon_indices:
                    count += 1
                    # if count % 2000 == 0:
                    #     progress = float(count)/total*100
                    #     print('\r{:.1f}%'.format(progress), end='')
                    # if not valid_func(vars, i, j, k):
                    if vars['nodata'][i][j][k]:
                        continue
                    table_row = table_class()
                    lat = vars['latitude'][j]
                    lon = vars['longitude'][k]
                    if (not lat or not lon
                        or lat == missing or lon == missing
                        or lat < min_lat or lat > max_lat
                        or lon < min_lon or lon > max_lon):
                        continue
                    # setattr(table_row, lat_name, float(lat))
                    # setattr(table_row, lon_name, float(lon))
                    table_row.latitude = float(lat)
                    table_row.longitude = float(lon)
                    # Set datetime
                    try:
                        mingmt = float(vars['mingmt'][i][j][k])
                        # See note about same mingmt for detail
                        if (mingmt == missing
                            or vars['mingmt'][0][j][k] == \
                            vars['mingmt'][1][j][k]):
                            continue
                        time_str ='{:02d}{:02d}00'.format(
                            *divmod(int(mingmt), 60))
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
                    except Exception as msg:
                        breakpoint()
                        exit(msg)
                    # Period check
                    if not datetime_ or not utils.check_period(datetime_,
                                                         self.period):
                        continue
                    table_row.datetime = datetime_

                    table_row.space_time = '%s %f %f' % (datetime_, lat, lon)

                    valid = True
                    table_row.land = bool(vars['land'][i][j][k])
                    table_row.ice = bool(vars['ice'][i][j][k])
                    if satel_name == 'ascat' or satel_name == 'qscat':
                        table_row.windspd = float(vars['windspd'][i][j][k])
                        table_row.winddir = float(vars['winddir'][i][j][k])
                        if (table_row.windspd is None 
                            or table_row.winddir is None
                            or table_row.windspd == missing
                            or table_row.winddir == missing):
                            continue
                        table_row.scatflag = float(vars['scatflag'][i][j][k])
                        table_row.radrain = float(vars['radrain'][i][j][k])
                        if satel_name == 'ascat':
                            table_row.sos = float(vars['sos'][i][j][k])
                    elif satel_name == 'wsat':
                        table_row.w_aw = float(vars['w-aw'][i][j][k])
                        table_row.wdir = float(vars['wdir'][i][j][k])
                        if (table_row.w_aw is None 
                            or table_row.wdir is None
                            or table_row.w_aw == missing
                            or table_row.wdir == missing):
                            continue
                        table_row.vapor = float(vars['vapor'][i][j][k])
                        table_row.cloud = float(vars['cloud'][i][j][k])
                        table_row.rain = float(vars['rain'][i][j][k])
                        table_row.w_lf = float(vars['w-lf'][i][j][k])
                        table_row.w_mf = float(vars['w-mf'][i][j][k])
                    else:
                        sys.exit('satel_name is wrong.')

                    if valid:
                        whole_table.append(table_row)

        return whole_table

    def _read_satel_dataset(self, satel_name, dataset, missing_val=-999.0):
        min_lat_index, max_lat_index = self._find_index(
            [self.region[0], self.region[1]], 'lat')
        lat_indices = [x for x in range(min_lat_index, max_lat_index+1)]
        min_lon_index, max_lon_index = self._find_index(
            [self.region[2], self.region[3]], 'lon')
        lon_indices = [x for x in range(min_lon_index, max_lon_index+1)]

    def _download_single_satel(self, config, satel_name, period):
        """Download ASCAT/QucikSCAT/Windsat data in specified date range.

        """
        info = config['prompt']['info']['download']
        self.logger.info(info)
        start_date = period[0].date()
        end_date = period[1].date()
        data_url = config['urls']
        file_suffix = config['data_suffix']
        save_dir = config['dirs']['bmaps']
        missing_dates_file = config['files_path']['missing_dates']

        utils.set_format_custom_text(config['data_name_length'])
        if os.path.exists(missing_dates_file):
            with open(missing_dates_file, 'rb') as fr:
                missing_dates = pickle.load(fr)
        else:
            missing_dates = set()

        os.makedirs(save_dir, exist_ok=True)
        delta_date = end_date - start_date

        total = delta_date.days + 1
        count = 0

        for i in range(delta_date.days + 1):
            count += 1
            print(f'\r({count}/{total})', end='')
            self.logger.debug(info)

            date_ = start_date + datetime.timedelta(days=i)
            if date_ in missing_dates:
                continue
            file_name = '%s_%04d%02d%02d%s' % (
                satel_name, date_.year, date_.month, date_.day, file_suffix)
            file_url = '%sy%04d/m%02d/%s' % (
                data_url, date_.year, date_.month, file_name)
            if not utils.url_exists(file_url):
                print('Missing date: ' + str(date_))
                print(file_url)
                missing_dates.add(date_)
                continue

            file_path = save_dir + file_name
            utils.download(file_url, file_path)
            self.downloaded_file_path[satel_name].append(file_path)

        utils.delete_last_lines()
        print('Done')

        with open(missing_dates_file, 'wb') as fw:
            pickle.dump(missing_dates, fw)

def datetime_from_bytemap(bm_file_path, vars, i, j, k, missing):
    bm_file_name = bm_file_path.split('/')[-1]
    date_str = bm_file_name.split('_')[1][:8]
    mingmt = int(vars['mingmt'][i][j][k])
    time_str = '{:02d}{:02d}00'.format(*divmod(mingmt, 60))

    datetime_ = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

    return datetime_

def valid_bytemap(vars, i, j, k):
    if vars['nodata'][i][j][k]:
        return False
    return True

def row2dict(row):
    d = row.__dict__
    d.pop('_sa_instance_state', None)

    return d
