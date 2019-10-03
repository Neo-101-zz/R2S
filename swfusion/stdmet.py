"""Manage downloading and reading NDBC Continuous Wind data.

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
from sqlalchemy import Integer, Float, String, DateTime
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy import tuple_
from sqlalchemy.orm import mapper
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import utils

n_ane = 0
n_ele = 0

Base = declarative_base()
DynamicBase = declarative_base(class_registry=dict())

class StdmetStation(Base):
    """Represents station which produces NDBC Continuous Wind data.

    """
    __tablename__ = 'stdmet_station'

    key = Column(Integer(), primary_key=True)
    id = Column(String(length=10), nullable=False, unique=True)
    type = Column(String(length=50))
    payload = Column(String(length=25))
    latitude = Column(Float(), nullable=False)
    longitude = Column(Float(), nullable=False)
    site_elev = Column(Float(), nullable=False)
    air_temp_elev = Column(Float())
    anemometer_elev = Column(Float(), nullable=False)
    barometer_elev = Column(Float())
    sea_temp_depth = Column(Float())
    water_depth = Column(Float())
    watch_circle_radius = Column(Float())

class StdmetManager(object):
    """Manage downloading and reading NDBC Continuous Wind data.

    """
    def __init__(self, CONFIG, period, region, passwd):
        self.STDMET_CONFIG = CONFIG['stdmet']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.stations = set()

        self.logger = logging.getLogger(__name__)

        self._analysis_and_save_relation()
        self._extract_relation()

        # self.download()
        # self.download_all_stations_no_limit()

        utils.reset_signal_handler()
        utils.setup_database(self, Base)

        # self.read()

        self.show()

    def download(self):
        """Download NDBC Continuous Wind data (including responding
        station's data).

        """
        correct, period = utils.check_and_update_period(
            self.period, self.STDMET_CONFIG['period_limit'],
            self.CONFIG['workflow']['prompt'])
        if not correct:
            return
        utils.setup_signal_handler()
        self._download_all_station_info()
        self._download_all_stdmet_data()

    def read(self, read_all=False):
        """Read data into MySQL database.

        """
        self._insert_station_info(read_all=True)
        self._insert_data()

        return

    def show(self):
        self._draw_stdmet_stations('focus')

    def _draw_stdmet_stations(self, mode='global'):
        stn_query = self.session.query(StdmetStation)
        stn_lat = []
        stn_lon = []

        for stn in stn_query:
            stn_lat.append(stn.latitude)
            stn_lon.append(stn.longitude)

        # setup Lambert Conformal basemap.
        if mode == 'global':
            map = Basemap(llcrnrlon=-300.0, llcrnrlat=-90.0, urcrnrlon=60.0,
                          urcrnrlat=90.0, projection='cyl')
        elif mode == 'focus':
            map = Basemap(llcrnrlon=min(stn_lon), llcrnrlat=min(stn_lat),
                          urcrnrlon=max(stn_lon), urcrnrlat=max(stn_lat),
                          projection='cyl')
        # draw coastlines.
        map.drawcoastlines()
        # draw a boundary around the map, fill the background.
        # this background will end up being the ocean color, since
        # the continents will be drawn on top.
        map.drawmapboundary(fill_color='aqua')
        # fill continents, set lake color same as ocean color.
        map.fillcontinents(color='coral',lake_color='aqua')
        map.drawmeridians(np.arange(0,360,30))
        map.drawparallels(np.arange(-90,90,30))
        map.scatter(stn_lon, stn_lat, latlon=True)
        plt.savefig((f'{self.CONFIG["result"]["dirs"]["fig"]}'
                     + f'distribution_of_stdmet_stations.png'))
        plt.show()

    def _create_stdmet_data_table(self, station_id):
        table_name = 'stdmet_%s' % station_id

        class StdmetData(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(StdmetData, t)

            return StdmetData

        cols = []
        cols.append(Column('key', Integer(), primary_key=True))
        cols.append(Column('date_time', DateTime(), nullable=False,
                           unique=True))
        cols.append(Column('wspd', Float(), nullable=False))
        cols.append(Column('wspd_10', Float(), nullable=False))
        cols.append(Column('wdir', Float(), nullable=False))
        cols.append(Column('gst', Float()))

        cols.append(Column('wvht', Float()))
        cols.append(Column('dpd', Float()))
        cols.append(Column('apd', Float()))
        cols.append(Column('mwd', Float()))
        cols.append(Column('pres', Float()))
        cols.append(Column('atmp', Float()))
        cols.append(Column('wtmp', Float()))
        cols.append(Column('dewp', Float()))
        cols.append(Column('vis', Float()))
        cols.append(Column('ptdy', Float()))
        cols.append(Column('tide', Float()))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(StdmetData, t)
        self.session.commit()

        return StdmetData

    def _insert_data(self, read_all=False):
        self.logger.info(self.STDMET_CONFIG['prompt']['info']['read_data'])
        data_dir = self.STDMET_CONFIG['dirs']['data']
        station_ids = [
            id for id in self.session.query(StdmetStation.id).\
            order_by(StdmetStation.id)
        ]
        if not read_all:
            data_files = [x for x in os.listdir(data_dir) if
                          x.endswith('.txt.gz')
                          and int(x[6:10]) in self.years]
        else:
            data_files = [x for x in os.listdir(data_dir) if
                          x.endswith('.txt.gz')]

        total = len(data_files)
        count = 0

        for id in station_ids:
            id = id[0]
            DataOfStation = self._create_stdmet_data_table(id)
            for file in data_files:
                if file.startswith(id):
                    # stdmet data file belong to station in stdmet_station
                    # table
                    count += 1
                    data_path = data_dir + file

                    info = f'Extracting stdmet data from {file}'
                    print(f'\r{info} ({count}/{total})', end='')

                    start = time.process_time()
                    records = self._extract_data(data_path, DataOfStation)
                    end = time.process_time()

                    self.logger.debug(f'{info} in {end-start:.2f} s')

                    if not records:
                        continue

                    start = time.process_time()
                    utils.bulk_insert_avoid_duplicate_unique(
                        records, int(self.CONFIG['database']\
                                     ['batch_size']['insert']/10),
                        DataOfStation, ['date_time'], self.session,
                        check_self=True)
                    end = time.process_time()

                    self.logger.debug((f'Bulk inserting stdmet data into '
                                       + f'stdmet_{id} in {end-start:2f} s'))
        utils.delete_last_lines()
        print('Done')

    def _insert_station_info(self, read_all=False):
        self.logger.info(self.STDMET_CONFIG['prompt']['info']\
                         ['read_station'])
        min_lat, max_lat = self.region[0], self.region[1]
        min_lon, max_lon = self.region[2], self.region[3]
        station_info_dir = self.STDMET_CONFIG['dirs']['stations']

        station_files = []
        if not read_all:
            for file in os.listdir(station_info_dir):
                if not file.endswith('.txt'):
                    continue
                for year in self.years:
                    for stn in self.year_station[year]:
                        if file == f'{stn}.txt':
                            station_files.append(file)
                            break
                    if file in station_files:
                        break
        else:
            station_files = [x for x in os.listdir(station_info_dir) if
                             x.endswith('.txt')]

        all_stations = []
        total = len(station_files)
        count = 0
        for file_name in station_files:
            count += 1
            station_info_path = station_info_dir + file_name

            info = f'Extracting station information from {file_name}'
            print((f'\r{info} ({count}/{total})'), end='')

            start = time.process_time()
            station = self._extract_station_info(station_info_path)
            end = time.process_time()

            self.logger.debug(f'{info} in {end-start:.2f} s')
            if station:
                all_stations.append(station)

        utils.delete_last_lines()
        print('Done')

        start = time.process_time()
        utils.bulk_insert_avoid_duplicate_unique(
            all_stations, self.CONFIG['database']['batch_size']['insert'],
            StdmetStation, ['id'], self.session)
        end = time.process_time()

        print('n_ane: ' + str(n_ane))
        print('n_ele: ' + str(n_ele))

        self.logger.debug(('Bulk inserting stdmet station information into '
                           + f'{StdmetStation.__tablename__} '
                           + f'in {end-start:.2f}s'))

    def _extract_data(self, data_path, DataOfStation):
        data_name = data_path.split('/')[-1]
        try:
            with gzip.GzipFile(data_path, 'rb') as gz:
                stdmet_text = gz.read()
        except FileNotFoundError as msg:
            exit(msg)
        except EOFError as msg:
            exit(msg + ': ' + data_path)

        temp_file_name = data_name[0:-3]
        with open(temp_file_name, 'wb') as txt:
            txt.write(stdmet_text)

        with open(temp_file_name, 'rb') as txt:
            first_line = txt.readline().decode('utf-8')

        names = ['year', 'month', 'day', 'hour']
        formats = ['i4', 'i2', 'i2', 'i2']
        if 'mm' in first_line:
            names.append('minute')
            formats.append('i2')
        names += ['wdir', 'wspd', 'gst', 'wvht', 'dpd' ,'apd', 'mwd',
                  'pres', 'atmp', 'wtmp', 'dewp', 'vis']
        formats += ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'i4', 'f4',
                    'f4', 'f4', 'f4', 'f4']
        if 'PTDY' in first_line:
            names.append('ptdy')
            formats.append('f4')
        if 'TIDE' in first_line:
            names.append('tide')
            formats.append('f4')

        # Specify data type of columns of unzipped gzip file
        data_type = {'names': tuple(names),
                     'formats': tuple(formats)}
        data = np.genfromtxt(temp_file_name, skip_header=1,
                             invalid_raise=False, dtype=data_type)
        os.remove(temp_file_name)

        # Store Continuous Wind Data in an entire year into a list
        stdmet_1_year = []

        if data[3]['year'] < 100:
            year_base = 1900
        else:
            year_base = 0

        for row in data:
            # watch datatype of each column
            # breakpoint()
            # Every row of data is the record of 10 minutes
            stdmet_1_hour = DataOfStation()
            row['year'] = year_base + row['year']
            if 'minute' in data_type['names']:
                minute = int(row['minute'])
            else:
                minute = 0
            stdmet_1_hour.datetime = datetime.datetime(
                int(row['year']), int(row['month']), int(row['day']),
                int(row['hour']), minute, 0)
            # When row's datetime is after period, skip all rows
            # left, because all rows are sorted in chronological
            # order
            if stdmet_1_hour.datetime > self.period[1]:
                break
            elif (stdmet_1_hour.datetime < self.period[0]
                  or stdmet_1_hour.datetime is None):
                continue
            anemometer_elev = self.session.query(StdmetStation).\
                    filter_by(id=data_name[:-12]).\
                    first().anemometer_elev
            if row['wspd'] == 99.0:
                continue
            else:
                stdmet_1_hour.wspd = float(row['wspd'])
            if stdmet_1_hour.wspd is None:
                continue
            stdmet_1_hour.wspd_10 = utils.convert_10(float(row['wspd']),
                                                     anemometer_elev)
            if row['wdir'] == 999:
                continue
            else:
                stdmet_1_hour.wdir = float(row['wdir'])
            if stdmet_1_hour.wdir is None:
                continue
            stdmet_1_hour.gst = (None if row['gst'] == 99.0 \
                                 else float(row['gst']))
            stdmet_1_hour.wvht = (None if row['wvht'] == 99.0 \
                                 else float(row['wvht']))
            stdmet_1_hour.dpd = (None if row['dpd'] == 99.0 \
                                 else float(row['dpd']))
            stdmet_1_hour.apd = (None if row['apd'] == 99.0 \
                                 else float(row['apd']))
            stdmet_1_hour.mwd= (None if row['mwd'] == 999 \
                                 else int(row['mwd']))
            stdmet_1_hour.pres = (None if row['pres'] == 9999.0 \
                                 else float(row['pres']))
            stdmet_1_hour.atmp = (None if row['atmp'] == 999.0 \
                                 else float(row['atmp']))
            stdmet_1_hour.wtmp = (None if row['wtmp'] == 999.0 \
                                 else float(row['wtmp']))
            stdmet_1_hour.dewp = (None if row['dewp'] == 999.0 \
                                 else float(row['dewp']))
            stdmet_1_hour.vis = (None if row['vis'] == 99.0 \
                                 else float(row['vis']))
            # Since field PTDY is not found in sample stdmet files,
            # PTDY will not be read for now
            if 'TIDE' in first_line:
                stdmet_1_hour.tide = (None if row['tide'] == 99.0 \
                                     else float(row['tide']))

            stdmet_1_year.append(stdmet_1_hour)

        if len(stdmet_1_year):
            return stdmet_1_year
        else:
            return None

    def _extract_station_info(self, station_info_path):
        station = StdmetStation()
        station.id = station_info_path.split('/')[-1][:-4]

        with open(station_info_path, 'r') as station_file:
            line_list = station_file.readlines()

        strict_pattern = "\d+\.\d+"
        min_lat, max_lat = self.region[0], self.region[1]
        min_lon, max_lon = self.region[2], self.region[3]

        for idx, line in enumerate(line_list):
            # Skip first line
            if not idx:
                continue
            elif ('buoy' in line or 'Station' in line
                  and station.type is None):
                station.type = line
            elif 'payload' in line and station.payload is None:
                station.payload = line.split('payload')[0][:-1]
            # Read latitude and longitude
            elif ('°' in line and '\'' in line and '"' in line
                  and station.latitude is None
                  and station.longitude is None):
                lat = float(re.findall(strict_pattern, line)[0])
                lon = float(re.findall(strict_pattern, line)[1])
                station.latitude = lat if 'N' in line else -lat
                station.longitude = lon if 'E' in line else 360. - lon
                if not (min_lat <= station.latitude <= max_lat
                        and min_lon <= station.longitude <= max_lon):
                    return None
            # Read elevation
            elif 'Site elevation' in line:
                station.site_elev = self._extract_elev(line)
            elif ('Air temp height' in line
                  and station.site_elev is not None):
                station.air_temp_elev = station.site_elev + \
                        self._extract_general_num(line)
            elif ('Anemometer height' in line 
                  and station.site_elev is not None):
                station.anemometer_elev = station.site_elev + \
                        self._extract_general_num(line)
            elif 'Barometer elevation' in line:
                station.barometer_elev = self._extract_elev(line)
            elif 'Sea temp depth' in line:
                station.sea_temp_depth = self._extract_general_num(line)
            elif 'Water depth' in line:
                station.water_depth = self._extract_general_num(line)
            elif 'Watch circle radius' in line:
                # Convert yard to meter
                station.watch_circle_radius = 0.9144 * \
                        self._extract_general_num(line)

        global n_ane, n_ele
        if station.anemometer_elev is None:
            n_ane += 1
        if station.site_elev is None:
            n_ele += 1

        if (station.anemometer_elev is None
            or station.site_elev is None
            or station.latitude is None
            or station.longitude is None):
            return None

        return station

    def _extract_general_num(self, line):
        pattern = r"[-+]?\d*\.\d+|\d+"
        return float(re.findall(pattern, line)[0])

    def _extract_elev(self, line):
        if ' m ' in line:
            num = self._extract_general_num(line)
            if 'above' in line:
                return num
            elif 'below' in line:
                return (-1) * num
        else:
            return 0

    def _extract_relation(self):
        """Then extract relation between inputted year(s) and station(s).

        """
        if not hasattr(self, 'all_year_station'):
            with open(self.STDMET_CONFIG['vars_path']['all_year_station'],
                      'rb') as file:
                self.all_year_station = pickle.load(file)

        # key: station, value: year
        self.station_year = dict()
        # key: year, value: station
        self.year_station = dict()

        # Collect self.stations' id according to self.years
        for year in self.years:
            stns = self.all_year_station[year]
            self.year_station[year] = stns
            self.stations.update(stns)
            for stn in stns:
                if not stn in self.station_year:
                    self.station_year[stn] = set()
                self.station_year[stn].add(year)

    def _analysis_and_save_relation(self):
        """Analysis and save relation between all years and stations
        from NDBC's Standard Meteorological Data webpage.

        """
        this_year = datetime.datetime.today().year

        if (os.path.exists(self.STDMET_CONFIG['vars_path']\
                           ['all_year_station'])
            and os.path.exists(self.STDMET_CONFIG['vars_path']\
                               ['all_station_year'])):
            relation_modified_datetime = dict()
            relation_modified_datetime['all_year_station'] = \
                    datetime.datetime.fromtimestamp(os.path.getmtime(
                        self.STDMET_CONFIG['vars_path']['all_year_station']))
            relation_modified_datetime['all_station_year'] = \
                    datetime.datetime.fromtimestamp(os.path.getmtime(
                        self.STDMET_CONFIG['vars_path']['all_station_year']))

            lastest_relation = True
            for key in relation_modified_datetime.keys():
                if relation_modified_datetime[key].year < this_year:
                    lastest_relation = False

            if lastest_relation:
                return

        self.all_year_station = dict()
        self.all_station_year = dict()

        start_year = self.STDMET_CONFIG['period_limit']['start'].year
        end_year = self.STDMET_CONFIG['period_limit']['end'].year
        if end_year > this_year:
            end_year = this_year
        self.all_years = [x for x in range(start_year, end_year+1)]
        self.all_stations = set()
        for year in self.all_years:
            info = f'Finding stations of year {year}'
            self.logger.debug(info)
            print(f'\r{info}', end='')

            stns = self._station_in_a_year(year)
            self.all_year_station[year] = stns
            self.all_stations.update(stns)

            for stn in stns:
                if not stn in self.all_station_year:
                    self.all_station_year[stn] = set()
                self.all_station_year[stn].add(year)
        utils.delete_last_lines()
        print('Done')

        # Save two dicts which store the relation between all years and
        # stations
        utils.save_relation(self.STDMET_CONFIG['vars_path']['all_year_station'],
                            self.all_year_station)
        utils.save_relation(self.STDMET_CONFIG['vars_path']['all_station_year'],
                            self.all_station_year)

    def download_all_stations_no_limit(self):
        # There are several stations which can be found in
        # https://www.ndbc.noaa.gov/data/historical/stdmet/
        # but do not have station page:
        # ['46a54', '42a02', '42otp', '42a03', '46a35', '47072', '32st2',
        # '51wh2', '41nt1', '41nt2', '51wh1', '32st1', '46074', '4h364',
        # 'a025w', '4h390', '4h361', 'q004w', '4h394', 'b040z', 'a002e',
        # 'et01z']
        if not hasattr(self, 'all_station_year'):
            with open(self.STDMET_CONFIG['vars_path']['all_station_year'],
                      'rb') as file:
                self.all_station_year = pickle.load(file)

        self.all_stations = set()
        for stn in self.all_station_year.keys():
            self.all_stations.add(stn)

        self.logger.info(self.STDMET_CONFIG['prompt']['info']\
                         ['download_all_station'])
        total = len(self.all_stations)
        count = 0
        for stn in self.all_stations:
            count += 1
            info = f'Downloading information of stdmet station {stn}'
            self.logger.debug(info)
            print((f'\r{info} ({count}/{total})'), end='')

            i = 0
            while True:
                # download self.stations' information
                result = self._download_single_station_info(stn)
                if result != 'error':
                    break
                else:
                    # Only loop when cannot get html of stdmet station
                    # webpage
                    self.logger.error(self.STDMET_CONFIG['prompt']['error'] \
                          ['fail_download_station'] + stn)
                    i += 1
                    if i <= self.STDMET_CONFIG['retry_times']:
                        self.logger.info('reconnect: %d' % i)
                    else:
                        self.logger.critical(
                            self.STDMET_CONFIG['prompt']['info']\
                            ['skip_download_station'])
                        break
        utils.delete_last_lines()
        print('Done')

    def _download_all_station_info(self):
        """Download all self.stations' information into single directory.

        """
        self.logger.info(self.STDMET_CONFIG['prompt']['info']\
                         ['download_station'])
        total = len(self.stations)
        count = 0
        for stn in self.stations:
            count += 1
            info = f'Downloading information of stdmet station {stn}'
            self.logger.debug(info)
            print((f'\r{info} ({count}/{total})'), end='')

            i = 0
            while True:
                # download self.stations' information
                result = self._download_single_station_info(stn)
                if result != 'error':
                    break
                else:
                    # Only loop when cannot get html of stdmet station
                    # webpage
                    self.logger.error(self.STDMET_CONFIG['prompt']['error'] \
                          ['fail_download_station'] + stn)
                    i += 1
                    if i <= self.STDMET_CONFIG['retry_times']:
                        self.logger.info('reconnect: %d' % i)
                    else:
                        self.logger.critical(
                            self.STDMET_CONFIG['prompt']['info']\
                            ['skip_download_station'])
                        break
        utils.delete_last_lines()
        print('Done')

    def _download_all_stdmet_data(self):
        """Download Continuous Wind data into single directory.

        """
        self.logger.info(self.STDMET_CONFIG['prompt']['info']\
                         ['download_data'])
        utils.set_format_custom_text(self.STDMET_CONFIG['data_name_length'])
        total = 0
        count = 0
        for year in self.years:
            total += len(self.year_station[year])

        for year in self.years:
            for stn in self.year_station[year]:
                self._download_single_stdmet_data(stn, year)
                count += 1
                info = f'Downloading {year} stdmet data of station {stn}'
                self.logger.debug(info)
                print((f'\r{info} ({count}/{total})'), end='')
        utils.delete_last_lines()
        print('Done')

    def _station_in_a_year(self, year):
        """Get stations' id in specified year.

        """
        url = self.STDMET_CONFIG['urls']['data']
        page = requests.get(url)
        data = page.text
        soup = BeautifulSoup(data, features='lxml')
        stations = set()
        suffix = 'h%s.txt.gz' % year
        anchors = soup.find_all('a')

        for link in anchors:
            href = link.get('href')
            if href.endswith(suffix):
                stn_id = href.split(suffix)[0]
                stations.add(stn_id)

        return stations

    def _download_single_station_info(self, station):
        """Download single stdmet station information.

        """
        url = self.STDMET_CONFIG['urls']['stations']
        save_dir = self.STDMET_CONFIG['dirs']['stations']
        file_name = station + '.txt'
        file_path = save_dir + file_name

        if os.path.exists(file_path):
            return True

        payload = dict()
        payload['station'] = str.lower(station)
        try:
            html = requests.get(url, params=payload, verify=True)
        except Exception as msg:
            self.logger.exception(('Exception occurred when getting '
                                   + 'html with requests'))
            return 'error'
        page = BeautifulSoup(html.text, features='lxml')
        div = page.find_all('div', id='stn_metadata')
        div = BeautifulSoup(str(div), features='lxml')
        information = div.find_all('p')
        if len(information) < 2:
            return False
        # write_information(file_path, information[1].text.replace('\xa0'*8, '\n\n'))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as stn_file:
            stn_file.write(information[1].text)

        return True

    def _download_single_stdmet_data(self, station, year):
        """Download Continuous Wind data of specified station and year.

        """
        save_dir = self.STDMET_CONFIG['dirs']['data']
        data_url = self.STDMET_CONFIG['urls']['data']
        os.makedirs(save_dir, exist_ok=True)
        file_name = '{0}h{1}.txt.gz'.format(station, year)
        file_path = '{0}{1}'.format(save_dir, file_name)
        file_url = '{0}{1}'.format(data_url, file_name)

        utils.download(file_url, file_path)
