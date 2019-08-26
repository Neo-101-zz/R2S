"""Manage downloading and reading NOAA HRD SFMR hurricane data.

"""
import datetime as dt
import gzip
import math
import pickle
import re
import os
import time

from bs4 import BeautifulSoup
import mysql.connector
import numpy as np
import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import tuple_
from sqlalchemy.schema import Table

import utils
import netcdf_util

MASKED = np.ma.core.masked
Base = declarative_base()
DynamicBase = declarative_base(class_registry=dict())

class SfmrManager(object):

    def __init__(self, CONFIG, period, region):
        self.SFMR_CONFIG = CONFIG['sfmr']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.years = None
        # self.download()
        self.read(read_all=True)

    def download(self):
        """Download SFMR data of hurricanes from NOAA HRD.

        """
        correct, period = utils.check_and_update_period(
            self.period, self.SFMR_CONFIG['period_limit'],
            self.CONFIG['workflow']['prompt'])
        if not correct:
            return
        utils.arrange_signal()
        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.year_hurr = self._gen_year_hurr()
        self._download_sfmr_data()

    def read(self, read_all=False):
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

        self._insert_sfmr(read_all)
        # self._insert_station_info(read_all)
        # st = time.time()
        # self._insert_data(read_all)
        # et = time.time()
        # print('Time: %s' % (et - st))


        return

    def _insert_sfmr(self, read_all):
        with open(self.SFMR_CONFIG['vars_path']['year_hurr'], 'rb') as fr:
            year_hurr = pickle.load(fr)

        data_root_dir = self.SFMR_CONFIG['dirs']['hurr']
        if read_all:
            files_path = []
            for year in year_hurr.keys():
                for hurr in year_hurr[year]:
                    spec_data_dir = '{0}{1}/{2}/'.format(
                        data_root_dir, year, hurr)
                    file_names = os.listdir(spec_data_dir)

                    for file in file_names:
                        if file.endswith('.nc'):
                            files_path.append(spec_data_dir + file)
        else:
            files_path = self._files_path_downloaded_in_this_run


        # Classify files_path by corresponding year
        files_years = set([x.split('/')[-3] for x in files_path])
        year_file_path = dict()
        for year in files_years:
            year_file_path[year] = []
            for file_path in files_path:
                if file_path.split('/')[-3] == year:
                    year_file_path[year].append(file_path)


        # Create SFMR table
        table_name_prefix = self.SFMR_CONFIG['table_names']['prefix']
        nc_template_path = files_path[0]
        skip_vars = ['DATE', 'TIME']
        notnull_vars = ['LAT', 'LON', 'SRR', 'SWS']
        unique_vars = []
        custom_cols = {1: Column('DATETIME', DateTime(),
                                 nullable=False, unique=False),
                       21: Column('SPACE_TIME', String(255),
                                 nullable=False, unique=True)}

        for year in year_file_path:
            table_name = table_name_prefix + year
            print((self.SFMR_CONFIG['prompt']['info']['insert_year_table'] \
                   + table_name))
            SfmrTable = utils.create_table_from_netcdf(
                self.engine, nc_template_path, table_name,
                self.session, skip_vars, notnull_vars, unique_vars,
                custom_cols)

            for file_path in year_file_path[year]:
                if not file_path.endswith('02H1.nc'):
                    continue
                hurr_name = file_path.split('/')[-2]
                print((self.SFMR_CONFIG['prompt']['info']['extract_netcdf']\
                       + hurr_name))
                print(file_path)
                one_day_records = self._extract_sfmr_from_netcdf(file_path,
                                                                 SfmrTable)

                total_sample = one_day_records
                batch_size = 1000
                table_class = SfmrTable
                unique_cols = ['SPACE_TIME']
                session = self.session
                utils.bulk_insert_avoid_duplicate_unique(
                    total_sample, batch_size, table_class, unique_cols,
                    session)

    def _extract_sfmr_from_netcdf(self, file_path, SfmrTable):
        """Dump one SFMR NetCDF file into one pickle file.

        """
        nc_file = file_path
        table_class = SfmrTable
        skip_vars = ['DATE', 'TIME']
        datetime_func = datetime_from_netcdf
        datetime_col_name = 'DATETIME'
        missing = MASKED
        valid_func = valid_netcdf
        unique_func = gen_space_time_fingerprint
        unique_col_name = 'SPACE_TIME'
        lat_name = 'LAT'
        lon_name = 'LON'
        period = self.period
        region = self.region
        not_null_vars = ['LAT', 'LON', 'SWS', 'SRR']

        res = utils.extract_netcdf_to_table(
            nc_file, table_class, skip_vars, datetime_func,
            datetime_col_name, missing, valid_func, unique_func,
            unique_col_name, lat_name, lon_name, period, region,
            not_null_vars)

        return res

    def _download_sfmr_data(self):
        """Download SFMR data of hurricanes.

        Parameters
        ----------
        None
            Nothing is required by this function.

        Returns
        -------
        hit_times : dict
            Times of hurricane NetCDF file's date being in period.

        """
        print(self.SFMR_CONFIG['prompt']['info']['download_hurr'])
        utils.set_format_custom_text(self.SFMR_CONFIG['data_name_length'])
        suffix = '.nc'
        save_root_dir = self.SFMR_CONFIG['dirs']['hurr']
        os.makedirs(save_root_dir, exist_ok=True)

        self._files_path_downloaded_in_this_run = []

        for year in self.year_hurr.keys():
            # Create directory to store SFMR files
            # os.makedirs('{0}{1}'.format(save_root_dir, year), exist_ok=True)
            hurrs = list(self.year_hurr[year])
            for hurr in hurrs:
                dir_path = '{0}{1}/{2}/'.format(save_root_dir, year, hurr)
                os.makedirs(dir_path, exist_ok=True)
                # Generate keyword to consist url
                keyword = '{0}{1}'.format(hurr, year)
                url = '{0}{1}{2}'.format(
                    self.SFMR_CONFIG['urls']['prefix'], keyword,
                    self.SFMR_CONFIG['urls']['suffix'])
                # Get page according to url
                page = requests.get(url)
                data = page.text
                soup = BeautifulSoup(data, features='lxml')
                anchors = soup.find_all('a')

                # Times of NetCDF file's date being in period
                for link in anchors:
                    href = link.get('href')
                    # Find href of netcdf file
                    if href.endswith(suffix):
                        # Extract file name
                        file_name = href.split('/')[-1]
                        date_str = file_name[-13:-5]
                        date_ = dt.date(int(date_str[:4]),
                                        int(date_str[4:6]),
                                        int(date_str[6:]))
                        if not utils.check_period(date_, self.period):
                            continue
                        file_path = dir_path + file_name
                        self._files_path_downloaded_in_this_run.append(file_path)
                        utils.download(href, file_path)

    def _gen_year_hurr(self):
        year_hurr = {}

        for year in self.years:
            if int(year) < 1994:
                year = 'prior1994'
            url = '{0}{1}.html'.format(
                self.SFMR_CONFIG['urls']['hurricane'][:-5], year)
            page = requests.get(url)
            data = page.text
            soup = BeautifulSoup(data, features='lxml')
            anchors = soup.find_all('a')

            year_hurr[year] = set()

            for link in anchors:
                if not link.contents:
                    continue
                text = link.contents[0]
                if text != 'SFMR':
                    continue
                href = link.get('href')
                hurr = href.split('/')[-2][:-4]
                year_hurr[year].add(hurr)

        utils.save_relation(self.SFMR_CONFIG['vars_path']['year_hurr'],
                            year_hurr)
        return year_hurr

def valid_netcdf(vars, index):
    if vars['FLAG'][index]:
        return False
    return True

def datetime_from_netcdf(vars, index, missing):
    """
    Note
    ----
    Only supports 'if var is missing' check now. Have not supported
    'if var == missing' check.

    """
    DATE, TIME = str(vars['DATE'][index]), str(vars['TIME'][index])
    # TIME variable's valid range is [0, 235959]
    if len(TIME) < 6:
        TIME = '0' * (6 - len(TIME)) + TIME
    if DATE is missing or TIME is missing:
        return False
    datetime_ = dt.datetime.strptime(DATE + TIME, '%Y%m%d%H%M%S')

    return datetime_

def gen_space_time_fingerprint(datetime, lat, lon):

    return '%s %f %f' % (datetime, lat, lon)
