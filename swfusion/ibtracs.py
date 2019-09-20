import datetime
import gzip
import logging
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
from netCDF4 import Dataset

import utils
import netcdf_util

MASKED = np.ma.core.masked
Base = declarative_base()

class WMOWPTC(Base):
    __tablename__ = 'tc_wp_wmo'

    key = Column(Integer, primary_key=True)
    sid = Column(String(13), nullable=False)
    datetime = Column(DateTime, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    pres = Column(Integer)
    wind = Column(Integer)
    sid_datetime = Column(String(50), nullable=False, unique=True)

class IBTrACS(object):
    def __init__(self, CONFIG, period, region, passwd):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        utils.setup_database(self, Base)
        self.download()
        self.read()

    def download(self):
        url = self.CONFIG['ibtracs']['urls']['wp']
        file = url.split('/')[-1]
        file = file[:-3].replace('.', '_') + '.nc'
        self.wp_file_path = f'{self.CONFIG["ibtracs"]["dirs"]["wp"]}{file}'
        os.makedirs(self.wp_file_path, exist_ok=True)

        utils.download(url, self.wp_file_path)

    def read(self):
        dataset = Dataset(self.wp_file_path)
        vars = dataset.variables
        wmo_pres = vars['wmo_pres']
        wmo_wind = vars['wmo_wind']

        if wmo_pres.shape != wmo_wind.shape:
            self.logger.error('shape of wmo_pres is not equal to wmo_wind')

        wmo_shape = wmo_pres.shape
        storm_num, date_time_num = wmo_shape[0], wmo_shape[1]
        all_wmo_wp_tcs = []

        total = storm_num * date_time_num
        count = 0
        info = f'Reading WMO records in {file_path.split("/")[-1]}'

        for i in range(storm_num):
            if int(vars['season'][i]) not in self.years:
                count += date_time_num
                continue
            sid = vars['sid'][i].tostring().decode('utf-8')

            for j in range(date_time_num):
                count += 1
                print(f'\r{info} {count}/{total}', end='')

                row = WMOWPTC()
                iso_time = vars['iso_time'][i][j]
                if iso_time[0] is MASKED:
                    continue
                iso_time_str = iso_time.tostring().decode('utf-8')
                row.datetime = datetime.datetime.strptime(
                    iso_time_str, '%Y-%m-%d %H:%M:%S')
                if not utils.check_period(row.datetime, self.period):
                    continue

                lat = vars['lat'][i][j]
                lon = vars['lon'][i][j]
                if lat is MASKED or lon is MASKED:
                    continue
                pres = vars['wmo_pres'][i][j]
                wind = vars['wmo_wind'][i][j]
                if pres is MASKED or wind is MASKED:
                    continue

                row.sid = sid
                row.lat = float(lat)
                row.lon = float(lon)
                row.pres = int(pres) if pres is not MASKED else None
                row.wind = int(wind) if wind is not MASKED else None
                row.sid_datetime = f'{sid}_{row.datetime}'

                all_wmo_wp_tcs.append(row)

        utils.bulk_insert_avoid_duplicate_unique(
            all_wmo_wp_tcs,
            self.CONFIG['database']['batch_size']['insert'],
            WMOWPTC, ['sid_datetime'], self.session,
            check_self=True)

        utils.delete_last_lines()
        print('Done')
