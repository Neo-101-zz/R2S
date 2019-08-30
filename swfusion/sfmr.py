"""Manage downloading and reading NOAA HRD SFMR hurricane data.

"""
import datetime
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
from sqlalchemy import Column, Integer, Float, String, DateTime, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy import tuple_
from sqlalchemy.schema import Table

import utils
import netcdf_util

MASKED = np.ma.core.masked
Base = declarative_base()
DynamicBase = declarative_base(class_registry=dict())

class HurrSfmr(Base):
    __tablename__ = 'hurr_sfmr'

    key = Column(Integer, primary_key=True)
    name = Column(String(length=20), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    min_lat = Column(Float, nullable=False)
    max_lat = Column(Float, nullable=False)
    min_lon = Column(Float, nullable=False)
    max_lon = Column(Float, nullable=False)
    name_period = Column(String(100), nullable=False, unique=True)

class SfmrManager(object):

    def __init__(self, CONFIG, period, region):
        self.SFMR_CONFIG = CONFIG['sfmr']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.years = None
        self.download()
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
        # Create table of SFMR records of hurricanes
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        self._insert_sfmr(read_all)

        return

    def _insert_sfmr(self, read_all):
        with open(self.SFMR_CONFIG['vars_path']['year_hurr'], 'rb') as fr:
            year_hurr = pickle.load(fr)

        data_root_dir = self.SFMR_CONFIG['dirs']['hurr']
        # if read_all:
        files_path = []
        all_hurr_records = []
        for year in year_hurr.keys():
            for hurr in year_hurr[year]:
                spec_data_dir = '{0}{1}/{2}/'.format(
                    data_root_dir, year, hurr)
                file_names = os.listdir(spec_data_dir)

                start_date = datetime.date(9999, 12, 31)
                end_date = datetime.date(1, 1, 1)
                nc_count = 0
                for file in file_names:
                    if file.endswith('.nc'):
                        nc_count += 1
                        files_path.append(spec_data_dir + file)

                        date_ = datetime.datetime.strptime(
                            file.split('SFMR')[1][:8]+'000000',
                            '%Y%m%d%H%M%S').date()
                        if date_ < start_date:
                            start_date = date_
                        if date_ > end_date:
                            end_date = date_

                # If is no NetCDF file in hurricane's directories,
                # it means that hurricane is not in period
                if not nc_count:
                    continue

                hurr_record = HurrSfmr()
                hurr_record.name = hurr
                hurr_record.start_date = start_date
                hurr_record.end_date = end_date
                hurr_record.min_lat = 90.0
                hurr_record.max_lat = -90.0
                hurr_record.min_lon = 360.0
                hurr_record.max_lon = 0.0
                hurr_record.name_period = '{0} {1} {2}'.format(
                    hurr, start_date, end_date)

                all_hurr_records.append(hurr_record)

        utils.bulk_insert_avoid_duplicate_unique(
            all_hurr_records, self.CONFIG['database']['batch_size'],
            HurrSfmr, ['name_period'], self.session)

        if not read_all:
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
                hurr_name = file_path.split('/')[-2]
                print((self.SFMR_CONFIG['prompt']['info']['extract_netcdf']\
                       + hurr_name))
                print(file_path)
                one_day_records, min_lat, max_lat, min_lon, max_lon = \
                        self._extract_sfmr_from_netcdf(file_path, SfmrTable)

                total_sample = one_day_records
                batch_size = 1000
                table_class = SfmrTable
                unique_cols = ['SPACE_TIME']
                session = self.session
                utils.bulk_insert_avoid_duplicate_unique(
                    total_sample, batch_size, table_class, unique_cols,
                    session)

                # Update SFMR records of hurricanes
                hurr = file_path.split('/')[-2]
                date_ = datetime.datetime.strptime(
                    file_path.split('/')[-1].split('SFMR')[1][:8]+'000000',
                    '%Y%m%d%H%M%S').date()
                hurr_query = self.session.query(HurrSfmr).\
                        filter(HurrSfmr.name == hurr).\
                        filter(HurrSfmr.start_date <= date_).\
                        filter(HurrSfmr.end_date >= date_)

                if hurr_query.count() > 1:
                    exit('[Error] Not unique hurricane SFMR record queried.')
                target = hurr_query.first()

                if min_lat < target.min_lat:
                    target.min_lat = min_lat
                if max_lat > target.max_lat:
                    target.max_lat = max_lat
                if min_lon < target.min_lon:
                    target.min_lon = min_lon 
                if max_lon > target.max_lon:
                    target.max_lon = max_lon

                self.session.commit()

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
        unique_func = utils.gen_space_time_fingerprint
        unique_col_name = 'SPACE_TIME'
        lat_name = 'LAT'
        lon_name = 'LON'
        period = self.period
        region = self.region
        not_null_vars = ['LAT', 'LON', 'SWS', 'SRR']

        res, min_lat, max_lat, min_lon, max_lon = \
                utils.extract_netcdf_to_table(
                    nc_file, table_class, skip_vars, datetime_func,
                    datetime_col_name, missing, valid_func, unique_func,
                    unique_col_name, lat_name, lon_name, period, region,
                    not_null_vars)

        return res, min_lat, max_lat, min_lon, max_lon

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
            hurrs = list(self.year_hurr[year])
            for hurr in hurrs:
                # Create directory to store SFMR files
                dir_path = f'{save_root_dir}{year}/{hurr}/'
                os.makedirs(dir_path, exist_ok=True)
                # Generate keyword to consist url
                keyword = f'{hurr}{year}'
                url = (f'{self.SFMR_CONFIG["urls"]["prefix"]}'
                       + f'{keyword}'
                       + f'{self.SFMR_CONFIG["urls"]["suffix"]}')
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
                        tail_half = file_name.split('SFMR')[1]
                        try:
                            if (tail_half.startswith('20')
                                or tail_half.startswith('199')):
                                date_str = tail_half[:8]
                                date_ = datetime.date(int(date_str[:4]),
                                                int(date_str[4:6]),
                                                int(date_str[6:]))
                            else:
                                date_str = tail_half[:6]
                                date_ = datetime.date(int(f'20{date_str[:2]}'),
                                                int(date_str[2:4]),
                                                int(date_str[4:]))
                                file_name = (f'{file_name.split("SFMR")[0]}20'
                                             + f'{file_name.split("SFMR")[1]}')
                        except Exception as msg:
                            breakpoint()
                            exit(msg)
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
    datetime_ = datetime.datetime.strptime(DATE + TIME, '%Y%m%d%H%M%S')

    return datetime_

