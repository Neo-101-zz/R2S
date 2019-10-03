"""Manage downloading and reading HWind data for RMS.

"""
import datetime
import gzip
import linecache
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
import shapefile
import sqlalchemy as sa
from sqlalchemy import create_engine, extract
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import Table, Column, MetaData
from sqlalchemy.orm import mapper
import matplotlib.pyplot as plt

import utils

Base = declarative_base()

class TCInfo(object):
    def __init__(self, year, basin, name):
        self.year = year
        self.basin = basin
        self.name = name

class HWindManager(object):
    """

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

        utils.setup_database(self, Base)
        # self.download()

        utils.reset_signal_handler()
        self.read()

    def get_hwind_class(self, sid, dt):
        dt_str = dt.strftime('%Y%_m%d_%H%M')
        table_name = (f'{self.CONFIG["hwind"]["table_name"]}'
                      + f'_{sid}_{dt_str}')

        class HWind(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(HWind, t)

            return table_name, None, HWind

        cols = []
        # IBTrACS columns
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))

        cols.append(Column('windspd', Float, nullable=False))
        cols.append(Column('winddir', Float, nullable=False))

        cols.append(Column('x_y', String(50), nullable=False,
                           unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        mapper(HWind, t)

        return table_name, t, HWind

    def download(self):
        utils.setup_signal_handler()
        self.no_data_count = dict()
        self.no_data_count['shapefile'] = 0
        self.no_data_count['gridded'] = 0

        self.year_tc = self._create_year_tc()

        self.logger.info(f'Downloading HWind data')

        total = 0
        count = 0
        for year in self.year_tc.keys():
            total += len(self.year_tc[year])

        for year in self.year_tc.keys():
            for tc in self.year_tc[year]:
                count += 1
                info = (f'Download HWind data of TC {tc.name} '
                        + f'in {year}')
                if count > 1:
                    utils.delete_last_lines()
                print(f'\r{info} ({count}/{total})', end='')

                for format in ['gridded']:
                    res = self._download_single_tc(
                        year, tc, self.CONFIG['hwind']['dirs'][format],
                        self.CONFIG['hwind']['data_link_text'][format])
                    if not res:
                        self.no_data_count[format] += 1

        utils.delete_last_lines()
        print('Done')
        print(self.no_data_count)

    def _download_single_tc(self, year, tc, tc_dir, data_link_text):
        url = self.CONFIG['hwind']['url']
        tc_url = f'{url}/{tc.name.lower()}{year}.html'
        tc_dir = f'{tc_dir}{tc.basin}/{year}/{tc.name}/'
        os.makedirs(tc_dir, exist_ok=True)

        page = requests.get(tc_url)
        data = page.text
        soup = BeautifulSoup(data, features='lxml')
        anchors = soup.find_all('a', text=data_link_text)
        if not len(anchors):
            return False

        url_prefix = self.CONFIG['hwind']['data_url_prefix']
        for a in anchors:
            href = a.get("href")
            file_name = href.split('/')[-1]
            file_url = f'{url_prefix}{href}'
            file_path = f'{tc_dir}{file_name}'

            if not utils.check_period(self._get_dt_of_hwind_file(href),
                                      self.period):
                continue

            utils.download(file_url, file_path)

        return True

    def _load_year_tc(self):
        year_tc_path = self.CONFIG['hwind']['pickle']['year_tc']
        if os.path.exists(year_tc_path):
            with open(year_tc_path, 'rb') as file:
                self.year_tc = pickle.load(file)

    def _create_year_tc(self):
        # Structure of dictionary year_tc
        # year - tc_info (basin, sid and name)

        # Return existing year_tc
        year_tc_path = self.CONFIG['hwind']['pickle']['year_tc']

        # Create year_tc
        year_tc = dict()
        url = self.CONFIG['hwind']['url']

        # Get page according to url
        page = requests.get(url)
        data = page.text
        soup = BeautifulSoup(data, features='lxml')
        mydivs = soup.find_all('div', class_='legacy-year')

        for year in self.years:
            year_tc[year] = []
            if year <= 1994:
                year_name = '1994 &amp; Earlier'
            elif year > 2013:
                continue
            else:
                year_name = str(year)
            try:
                div_year = soup.find_all('div', class_='legacy-year',
                                   text=year_name)[0]
            except Exception as msg:
                self.logger.error((f'Error occurs when extract year div '
                                   + f'from {url}'))
            strong_basins = div_year.find_parent('div', class_='row').\
                    find_next_sibling('div').find_all('strong')
            for item in strong_basins:
                basin = self.CONFIG['hwind']['basin_map'][item.text]
                anchors = item.find_parent('div', class_='col-sm-4').\
                        find_all('a')
                for a in anchors:
                    tc_url = a.get('href')
                    tc_page = requests.get(tc_url)
                    tc_data = tc_page.text
                    tc_soup = BeautifulSoup(tc_data, features='lxml')

                    gridded_anchors = tc_soup.find_all(
                        'a', text=self.CONFIG['hwind']['data_link_text']\
                        ['gridded'])
                    if not len(gridded_anchors):
                        continue

                    url_prefix = self.CONFIG['hwind']['data_url_prefix']
                    tc_data_in_period_count = 0
                    for g_a in gridded_anchors:
                        if utils.check_period(self._get_dt_of_hwind_file(
                            g_a.get('href')), self.period):
                            tc_data_in_period_count += 1

                    if not tc_data_in_period_count:
                        continue

                    tc_info = TCInfo(year, basin, a.text)
                    self._find_sid(tc_info)

                    year_tc[year].append(tc_info)

        os.makedirs(os.path.dirname(year_tc_path), exist_ok=True)
        with open(year_tc_path, 'wb') as file:
            pickle.dump(year_tc, file)

        return year_tc

    def _find_sid(self, tc_info):
        # Get TC table and count its row number
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        tc_query = self.session.query(TCTable).\
                filter(extract('year', TCTable.date_time) == tc_info.year).\
                filter(TCTable.basin == tc_info.basin).\
                filter(TCTable.name == tc_info.name.upper())
        if not tc_query.count():
            print((f'SID not found: {tc_info.year} {tc_info.basin} '
                   + f'{tc_info.name}'))
            return

        tc_info.sid = tc_query.first().sid

    def _get_gridded_path(self, year, tc):
        root_dir = self.CONFIG['hwind']['dirs']['gridded']
        tc_dir = f'{root_dir}{tc.basin}/{tc.year}/{tc.name}/'
        gridded_paths = [f'{tc_dir}{x}' for x in os.listdir(tc_dir) if
                         x.endswith('.gz')]

        return gridded_paths

    def _get_tc_center(self, data_path):
        try:
            with gzip.GzipFile(data_path, 'rb') as gz:
                hwind_text = gz.read()
        except FileNotFoundError as msg:
            exit(msg)
        except EOFError as msg:
            exit(msg + ': ' + data_path)

        data_name = data_path.split('/')[-1]
        temp_file_name = data_name[0:-3]
        with open(temp_file_name, 'wb') as txt:
            txt.write(hwind_text)

        # Example: 'STORM CENTER LOCALE IS -95.3800 EAST LONGITUDE '
        # 'and  28.7490 NORTH LATITUDE ... STORM CENTER IS AT (X,Y)=(0,0)\n'
        line = linecache.getline(temp_file_name, 3)
        os.remove(temp_file_name)

        if 'EAST LONGITUDE' not in line or 'NORTH LATITUDE' not in line:
            self.logger.error((f'Find HWind gridded file whose center '
                               + f'is not represented in EAST LONGITUDE '
                               + f'and NORTH LATITUDE: {data_path}'))
        pattern = r"[-+]?\d*\.\d+|\d+"
        res = re.findall(pattern, line)
        lon, lat = float(res[0]), float(res[1])
        x, y = float(res[2]), float(res[3])
        lon = (lon + 360) % 360

        return lon, lat, x, y

    def _get_dt_of_hwind_file(self, gridded_file):
        file_name = gridded_file.split('/')[-1]
        datetime_str = file_name[4:18]
        try:
            dt = datetime.datetime.strptime(datetime_str, '%Y_%m%d_%H%M')
        except Exception as msg:
            self.logger.error((f'Error occurs when converting file path '
                               + f'to datetime: {gridded_file}'))

        return dt

    def _read_tc_gridded(self, data_path, hwind_table):
        tc_data = []

        try:
            with gzip.GzipFile(data_path, 'rb') as gz:
                hwind_text = gz.read()
        except FileNotFoundError as msg:
            exit(msg)
        except EOFError as msg:
            exit(msg + ': ' + data_path)

        data_name = data_path.split('/')[-1]
        temp_file_name = data_name[0:-3]
        with open(temp_file_name, 'wb') as txt:
            txt.write(hwind_text)

        with open(temp_file_name, 'r') as file:
            line_list = file.readlines()
        os.remove(temp_file_name)

        num_pattern = r"[-+]?\d*\.\d+|\d+"
        lons_line_idx = dict()
        lats_line_idx = dict()
        for idx, line in enumerate(line_list):
            if 'EAST LONGITUDE COORDINATES' in line:
                lons_line_idx['start'] = idx + 2
                lons_num = int(re.findall(num_pattern, line_list[idx+1])[0])

            if 'NORTH LATITUDE COORDINATES' in line:
                lons_line_idx['end'] = idx - 1
                lats_line_idx['start'] = idx + 2
                lats_num = int(re.findall(num_pattern, line_list[idx+1])[0])

            if 'SURFACE WIND COMPONENTS' in line:
                lats_line_idx['end'] = idx - 1
                wind_line_idx = idx + 2

        lons = []
        for i in range(lons_line_idx['start'], lons_line_idx['end']+1):
            lons_in_a_line = re.findall(num_pattern, line_list[i])
            for lon in lons_in_a_line:
                lons.append(float(lon))

        lats = []
        for i in range(lats_line_idx['start'], lats_line_idx['end']+1):
            lats_in_a_line = re.findall(num_pattern, line_list[i])
            for lat in lats_in_a_line:
                lats.append(float(lat))

        windspd = np.ndarray(shape=(lats_num, lons_num), dtype=float)
        winddir = np.ndarray(shape=(lats_num, lons_num), dtype=float)
        lon_idx = 0
        lat_idx = 0

        for i in range(wind_line_idx, len(line_list)):
            uv_wind_in_a_line = re.findall(num_pattern, line_list[i])
            for idx, value in enumerate(uv_wind_in_a_line):
                # U component
                if not idx % 2:
                    u_wind = float(value)
                # V component
                if idx % 2:
                    v_wind = float(value)
                    windspd[lat_idx][lon_idx] = math.sqrt(
                        u_wind**2 + v_wind**2) * 1.94384
                    winddir[lat_idx][lon_idx] = math.degrees(
                        math.atan2(u_wind, v_wind))
                    lon_idx += 1
                    if lon_idx == lons_num:
                        lon_idx = 0
                        lat_idx += 1

        for x in range(lons_num):
            for y in range(lats_num):
                row = hwind_table()
                row.x = x
                row.y = y
                row.x_y = f'{x}_{y}'
                row.lon = lons[x]
                row.lat = lats[y]
                row.windspd = float(windspd[y][x])
                row.winddir = float(winddir[y][x])

                tc_data.append(row)

        # Plot windspd in knots with matplotlib's contour
        # X, Y = np.meshgrid(lons, lats)
        # Z = windspd
        # fig, ax = plt.subplots()
        # CS = ax.contour(X, Y, Z, levels=[5,10,15,20,25])
        # ax.clabel(CS, inline=1, fontsize=10)
        # plt.show()
        # breakpoint()

        return tc_data

    def read(self):
        self._load_year_tc()

        for year in self.year_tc.keys():
            for tc in self.year_tc[year]:
                # Get gridded file_path
                gridded_paths = self._get_gridded_path(year, tc)
                for gridded_file in gridded_paths:
                    # Get TC center locale
                    lon, lat, x, y = self._get_tc_center(gridded_file)
                    dt = self._get_dt_of_hwind_file(gridded_file)
                    # Get HWind table
                    table_name, sa_table, hwind_table= self.get_hwind_class(
                        tc.sid, dt)

                    # Read shapefile
                    data = self._read_tc_gridded(gridded_file, hwind_table)

                    # Skip this turn of loop if not getting data matrix
                    if not len(data):
                        continue

                    # When ERA5 table doesn't exists, sa_table is None.
                    # So need to create it.
                    if sa_table is not None:
                        # Create table of ERA5 data cube
                        sa_table.create(self.engine)
                        self.session.commit()

                    # Insert into HWind table
                    start = time.process_time()
                    utils.bulk_insert_avoid_duplicate_unique(
                        data,
                        int(self.CONFIG['database']['batch_size']['insert']/10),
                        hwind_table, ['x_y'], self.session,
                        check_self=True)
                    end = time.process_time()

                    self.logger.debug((f'Bulk inserting HWind data into '
                                       + f'{table_name} in {end-start:2f} s'))
        utils.delete_last_lines()
        print('Done')
