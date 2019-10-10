"""Download and read IBTrACS.
https://www.ncdc.noaa.gov/ibtracs/index.php

"""
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
from sqlalchemy.schema import Table, MetaData
from netCDF4 import Dataset
from sqlalchemy.orm import mapper

import utils
import netcdf_util

MASKED = np.ma.core.masked
Base = declarative_base()

class IBTrACSManager(object):
    """Manage features of IBTrACS that are not related to other data
    sources.

    """
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
        self.download()

        utils.reset_signal_handler()
        utils.setup_database(self, Base)
        self.read()

    def create_tc_table(self):
        """Create the table which represents TC records from IBTrACS.

        """
        table_name = self.CONFIG['ibtracs']['table_name']

        class WMOWPTC(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(WMOWPTC, t)

            return WMOWPTC

        cols = []
        # IBTrACS columns
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('sid', String(13), nullable=False))
        cols.append(Column('name', String(50)))
        cols.append(Column('date_time', DateTime, nullable=False))
        cols.append(Column('basin', String(2), nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('pres', Integer))
        cols.append(Column('wind', Integer))

        cols.append(Column('r34_ne', Integer))
        cols.append(Column('r34_se', Integer))
        cols.append(Column('r34_sw', Integer))
        cols.append(Column('r34_nw', Integer))

        cols.append(Column('r50_ne', Integer))
        cols.append(Column('r50_se', Integer))
        cols.append(Column('r50_sw', Integer))
        cols.append(Column('r50_nw', Integer))

        cols.append(Column('r64_ne', Integer))
        cols.append(Column('r64_se', Integer))
        cols.append(Column('r64_sw', Integer))
        cols.append(Column('r64_nw', Integer))

        cols.append(Column('sid_date_time', String(50), nullable=False,
                           unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(WMOWPTC, t)
        self.session.commit()

        return WMOWPTC

    def download(self):
        """Download IBTrACS data.

        """
        self.logger.info('Downloading IBTrACS')
        utils.setup_signal_handler()
        utils.set_format_custom_text(
            self.CONFIG['ibtracs']['data_name_length'])

        url = self.CONFIG['ibtracs']['urls']['since1980']
        file = url.split('/')[-1]
        file = file[:-3].replace('.', '_') + '.nc'
        dir = self.CONFIG['ibtracs']['dirs']
        os.makedirs(dir, exist_ok=True)
        self.wp_file_path = f'{dir}{file}'

        utils.download(url, self.wp_file_path, progress=True)

    def read(self):
        """Read IBTrACS data into TC table.

        """
        self.logger.info('Reading IBTrACS')
        # Read NetCDF file of IBTrACS
        dataset = Dataset(self.wp_file_path)
        vars = dataset.variables
        # Use data from official WMO agency
        wmo_pres = vars['wmo_pres']
        wmo_wind = vars['wmo_wind']

        if wmo_pres.shape != wmo_wind.shape:
            self.logger.error('shape of wmo_pres is not equal to wmo_wind')

        wmo_shape = wmo_pres.shape
        # Get two dimensions of IBTrACS data
        storm_num, date_time_num = wmo_shape[0], wmo_shape[1]

        # Record whether each month in each year has been read
        have_read = dict()
        for year in self.years:
            have_read[year] = dict()
            for m in range(12):
                have_read[year][m+1] = False

        info = f'Reading WMO records in {self.wp_file_path.split("/")[-1]}'
        # Read detail of IBTrACS data
        self._read_detail(vars, storm_num, date_time_num, have_read, info)

    def _read_detail(self, vars, storm_num, date_time_num, have_read, info):
        """Read detail of IBTrACS data.

        """
        total = storm_num * date_time_num
        count = 0
        # List to record all details
        wmo_wp_tcs = []
        WMOWPTC = self.create_tc_table()

        for i in range(storm_num):
            skip_storm = False
            # Skip this loop if season (year) of TC not in period
            if int(vars['season'][i]) not in self.years:
                skip_storm = True
            if not skip_storm:
                # Skip this loop is datetime of first record is earlier than
                # start date of period of more than 60 days,
                # or datetime of first record is later than end date of period
                first_iso_time = vars['iso_time'][i][0]
                if first_iso_time[0] is MASKED:
                    continue
                first_datetime = datetime.datetime.strptime(
                    first_iso_time.tostring().decode('utf-8'),
                    '%Y-%m-%d %H:%M:%S')
                if first_datetime < (self.period[0] -
                                     datetime.timedelta(days=45)):
                    skip_strom = True
                elif first_datetime > self.period[1]:
                    skip_strom = True

            if skip_storm:
                count += date_time_num
                self.logger.debug((f'Skipping No.{i+1} TC in '
                                  + f'season {vars["season"][i]}'))
                continue

            self.logger.debug((f'Reading No.{i+1} TC in '
                              + f'season {vars["season"][i]}'))

            sid = vars['sid'][i].tostring().decode('utf-8')
            name = vars['name'][i]
            name = name[name.mask == False].tostring().decode('utf-8')

            for j in range(date_time_num):
                count += 1
                print(f'\r{info} {count}/{total}', end='')

                row = WMOWPTC()

                # Read ISO time and check whether record is in period
                iso_time = vars['iso_time'][i][j]
                if iso_time[0] is MASKED:
                    break

                iso_time_str = iso_time.tostring().decode('utf-8')
                row.date_time = datetime.datetime.strptime(
                    iso_time_str, '%Y-%m-%d %H:%M:%S')
                if not utils.check_period(row.date_time, self.period):
                    continue

                # Insert rows which have read to TC table until
                # find next unread month
                # year, month = row.date_time.year, row.date_time.month
                # if not have_read[year][month]:
                #     if len(wmo_wp_tcs):
                #         utils.bulk_insert_avoid_duplicate_unique(
                #             wmo_wp_tcs, self.CONFIG['database']\
                #             ['batch_size']['insert'],
                #             WMOWPTC, ['sid_date_time'], self.session,
                #             check_self=True)
                #         wmo_wp_tcs = []
                #     self.logger.debug((f'Reading WMO records of '
                #                       + f'{year}-{str(month).zfill(2)}'))
                #     have_read[year][month] = True

                # Read basin of TC
                row.basin = vars['basin'][i][j].tostring().decode('utf-8')

                # Read latitude, longitude, minimal centeral pressure,
                # maximum sustained wind speed from official WMO agency
                lat = vars['lat'][i][j]
                lon = vars['lon'][i][j]
                if lat is MASKED or lon is MASKED:
                    continue
                pres = vars['wmo_pres'][i][j]
                wind = vars['wmo_wind'][i][j]
                if pres is MASKED or wind is MASKED:
                    continue

                # Set attributes of row
                row.sid = sid
                if name != 'NOT_NAMED':
                    row.name = name
                row.lat = float(lat)
                row.lon = float(lon)
                row.pres = int(pres) if pres is not MASKED else None
                row.wind = int(wind) if wind is not MASKED else None
                row.sid_date_time = f'{sid}_{row.date_time}'

                # Average radius of 34/50/64 knot winds in four directoins
                # (ne, se, sw, nw) from three agencies (bom, reunion, usa)
                dirs = ['ne', 'se', 'sw', 'nw']
                radii = dict()
                for r in ['r34', 'r50', 'r64']:
                    radii[r] = dict()
                    for d in range(4):
                        radii[r][d] = []
                        for a in ['bom', 'reunion', 'usa']:
                            r_d_a = vars[f'{a}_{r}'][i][j][d]
                            if r_d_a is not MASKED:
                                radii[r][d].append(int(r_d_a))
                        if len(radii[r][d]):
                            setattr(row, f'{r}_{dirs[d]}',
                                    int(sum(radii[r][d])/len(radii[r][d])))

                wmo_wp_tcs.append(row)

        if len(wmo_wp_tcs):
            utils.bulk_insert_avoid_duplicate_unique(
                wmo_wp_tcs, self.CONFIG['database']\
                ['batch_size']['insert'],
                WMOWPTC, ['sid_date_time'], self.session,
                check_self=True)

        utils.delete_last_lines()
        print('Done')
