"""Manage downloading and reading ASCAT, QucikSCAT and Windsat data.

"""
import datetime
import math
import pickle
import os

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

    def __init__(self, CONFIG, period, region):
        self.satel_names = ['ascat', 'qscat', 'wsat']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region

        # self.download()
        self.read(read_all=True)

    def download(self):
        utils.arrange_signal()
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
        DB_CONFIG = self.CONFIG['database']
        PROMPT = self.CONFIG['workflow']['prompt']
        DBAPI = DB_CONFIG['db_api']
        USER = DB_CONFIG['user']
        # password_ = input(PROMPT['input']['db_root_password'])
        password_ = '39cnj971hw-'
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
                files_path[satel_name] += \
                        os.listdir(self.CONFIG[satel_name]['dirs']['bmaps'])
        else:
            files_path = self.downloaded_file_path

        for satel_name in files_path.keys():
            # Create table of particular satellite
            bytemap_file = files_path[satel_name][0]
            table_name = satel_name
            skip_vars = ['mingmt']
            notnull_vars = ['latitude', 'longitude']
            if satel_name == 'ascat' or satel_name == 'qscat':
                notnull_vars += ['windspd', 'winddir']
            elif satel_name == 'wsat':
                notnull_vars += ['w-aw', 'wdir'] 
            unique_vars = []
            custom_cols = {1: Column('DATETIME', DateTime(),
                                     nullable=False, unique=False),
                           -1: Column('SPACE_TIME', String(255),
                                     nullable=False, unique=True)}

            SatelTable = utils.create_table_from_bytemap(
                self.engine, satel_name, bytemap_file,
                table_name, self.session, skip_vars, notnull_vars,
                unique_vars, custom_cols)
            breakpoint()

            for file_path in files_path[satel_name]:
                    dataset = self._dataset_of_daily_satel(satel_name,
                                                           file_path)
                    one_day_records = self._read_satel_dataset(satel_name,
                                                               dataset)

    def _read_satel_dataset(self, satel_name, dataset, missing_val=-999.0):
        min_lat_index, max_lat_index = self._find_index(
            [self.region[0], self.region[1]], 'lat')
        lat_indices = [x for x in range(min_lat_index, max_lat_index+1)]
        min_lon_index, max_lon_index = self._find_index(
            [self.region[2], self.region[3]], 'lon')
        lon_indices = [x for x in range(min_lon_index, max_lon_index+1)]


    def _find_index(self, range, lat_or_lon):
        # latitude: from -89.875 to 89.875, 720 values, interval = 0.25
        # longitude: from 0.125 to 359.875, 1440 values, interval = 0.25
        res = []
        for idx, val in enumerate(range):
            if lat_or_lon == 'lat':
                delta = val + 89.875
            elif lat_or_lon == 'lon':
                delta = val - 0.125
            else:
                exit('Error parameter lat_or_lon: ' + lat_or_lon)
            intervals = delta / 0.25
            # Find index of min_lat or min_lon
            if not idx:
                res.append(math.ceil(intervals))
            # Find index of max_lat or max_lon
            else:
                res.append(math.floor(intervals))

        return res

    def _dataset_of_daily_satel(self, satel_name, file_path,
                                missing_val=-999.0):
        if satel_name == 'ascat':
            dataset = ASCATDaily(file_path, missing=missing_val)
        elif satel_name == 'qscat':
            dataset = QuikScatDaily(file_path, missing=missing_val)
        elif satel_name == 'wsat':
            dataset = WindSatDaily(file_path, missing=missing_val)
        else:
            sys.exit('Invalid satellite name')

        if not dataset.variables:
            sys.exit('[Error] File not found: ' + file_path)

        return dataset

    def _download_single_satel(self, config, satel_name, period):
        """Download ASCAT/QucikSCAT/Windsat data in specified date range.

        """
        print(config['prompt']['info']['download'])
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
        for i in range(delta_date.days + 1):
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

        with open(missing_dates_file, 'wb') as fw:
            pickle.dump(missing_dates, fw)
