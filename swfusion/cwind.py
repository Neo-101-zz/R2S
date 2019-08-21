"""Manage downloading and reading NDBC Continuous Wind data.

"""
import re
import os

import requests
from bs4 import BeautifulSoup
import sqlalchemy as sa
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import tuple_

import utils
import terminalsize

Base = declarative_base()

class CwindStation(Base):
    """Represents station which produces NDBC Continuous Wind data.

    """
    __tablename__ = 'cwind_station'

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

class CwindManager(object):
    """Manage downloading and reading NDBC Continuous Wind data.

    """
    def __init__(self, CONFIG, period, region):
        self.CWIND_CONFIG = CONFIG['cwind']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.years = None
        self.stations = None
        self.year_station = None
        self.station_year = None
        self.cwind_station_file_names = []
        self.cwind_data_file_names = []
        self.download()
        self.read()

    def download(self):
        """Download NDBC Continuous Wind data (including responding
        station's data).

        """
        correct, period = utils.check_period(
            self.period, self.CWIND_CONFIG['period_limit'],
            self.CONFIG['workflow']['prompt'])
        if not correct:
            return
        utils.arrange_signal()
        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.stations = set()
        self._analysis_and_save_relation()
        self._download_all_station_info()
        self._download_all_cwind_data()

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
        self.engine = create_engine(connect_string, echo=True)
        # Create table of cwind station
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        self._insert_station_info(all)

        return

    def _get_cwind_data_table_of_one_station(self, station_id):
        DynamicBase = declarative_base(class_registry=dict())

        class CwindData(DynamicBase):
            """Represents NDBC Continuous Wind data.

            """
            __tablename__ = 'cwind_%s' % station_id

            key = Column(Integer(), primary_key=True)
            datetime = Column(DateTime(), nullable=False, unique=True)
            wspd = Column(Float(), nullable=False)
            wdir = Column(Float(), nullable=False)
            gst = Column(Float())
            gdr = Column(Float())
            gtime = Column(Float())

        return CwindData

    def _insert_data(self, all):
        data_dir = self.CWIND_CONFIG['dirs']['data']
        station_ids = [
            id for id in self.session.query(CwindStation.id).\
            order_by(CwindStation.id)
        ]
        if all:
            data_files = [x for x in os.list(data_dir) if
                          x.endswith('.txt.gz')]
        else:
            data_files = self.cwind_data_file_names

        for id in station_ids:
            DataOfStation = \
                    _get_cwind_data_table_of_one_station(id)
            station_records = []
            for file in data_files:
                if file.startswith(id):
                    # cwind data file belong to station in cwind_station
                    # table
                    data_path = data_dir + file
                    records = self._extract_data(self, data_path,
                                                 DataOfStation)
                    if records:
                        station_records.append(records)

            # Choose out stations which have not been inserted into table
            while station_records:
                batch = station_records[0:1000]
                station_records = station_records[1000:]

                existing_records = dict(
                    (
                        (data.datetime),
                        data
                    )
                    for data in self.session.query(DataOfStation).filter(
                        tuple_(DataOfStation.datetime).in_(
                            [tuple_(x.datatime) for x in batch]
                        )
                    )
                )
                inserts = []
                for data in batch:
                    existing = existing_records.get(data.datetime, None)
                    if existing:
                        pass
                    else:
                        inserts.append(data)
                if inserts:
                    self.session.add_all(inserts)

            self.session.new
            self.session.commit()




    def _insert_station_info(self, all):
        min_lat, max_lat = self.region[0], self.region[1]
        min_lon, max_lon = self.region[2], self.region[3]
        station_info_dir = self.CWIND_CONFIG['dirs']['stations']
        if all:
            station_files = [x for x in os.listdir(station_info_dir) if
                             x.endswith('.txt')]
        else:
            station_files = self.cwind_station_file_names
        print(len(station_files))
        print(len(self.cwind_station_file_names))
        all_stations = []
        for file_name in station_files:
            station_info_path = station_info_dir + file_name
            station = self._extract_station_info(station_info_path)
            if station:
                all_stations.append(station)

        # Choose out stations which have not been inserted into table
        while all_stations:
            batch = all_stations[0:100]
            all_stations = all_stations[100:]

            existing_stations = dict(
                (
                    (stn.id),
                    stn
                )
                for stn in self.session.query(CwindStation).filter(
                    tuple_(CwindStation.id).in_(
                        [tuple_(x.id) for x in batch]
                    )
                )
            )
            inserts = []
            for stn in batch:
                existing = existing_stations.get(stn.id, None)
                if existing:
                    pass
                else:
                    inserts.append(stn)
            if inserts:
                self.session.add_all(inserts)

        self.session.new
        self.session.commit()

    def _extract_data(self, data_path, DataOfStation):
        pass


    def _extract_station_info(self, station_info_path):
        station = CwindStation()
        station.id = station_info_path.split('/')[-1][:5]

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

        if (station.anemometer_elev is None
            or station.site_elev is None
            or station.latitude is None
            or station.longitude is None):
            print(station.id)
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

    def _analysis_and_save_relation(self):
        """Analysis and save relation between inputted year(s) and station(s)
        from NDBC's Continuous Wind Historical Data webpage.

        """
        # key: station, value: year
        self.station_year = dict()
        # key: year, value: station
        self.year_station = dict()
        # Collect self.stations' id according to self.years
        for year in self.years:
            stns = self._station_in_a_year(year)
            self.year_station[year] = stns
            self.stations.update(stns)
            for stn in stns:
                if not stn in self.station_year:
                    self.station_year[stn] = set()
                self.station_year[stn].add(year)
        # Save two dicts which store the relation between self.stations and year
        utils.save_relation(self.CWIND_CONFIG['vars_path']['year_station'],
                            self.year_station)
        utils.save_relation(self.CWIND_CONFIG['vars_path']['station_year'],
                            self.station_year)

    def _download_all_station_info(self):
        """Download all self.stations' information into single directory.

        """
        print(self.CWIND_CONFIG['prompt']['info']['dl_cwind_station'])
        for stn in self.stations:
            i = 0
            while True:
                # download self.stations' information
                result = self._download_single_station_info(stn)
                if result != 'error':
                    break
                else:
                    # Only loop when cannot get html of cwind station
                    # webpage
                    print(self.CWIND_CONFIG['prompt']['error'] \
                          ['fail_dl_cwind_station'] + stn)
                    i += 1
                    if i <= self.CWIND_CONFIG['retry_times']:
                        print('reconnect: %d' % i)
                    else:
                        print(self.CWIND_CONFIG['prompt']['info'][
                            'skip_dl_cwind_station'])
                        break
        print()

    def _download_all_cwind_data(self):
        """Download Continuous Wind data into single directory.

        """
        print(self.CWIND_CONFIG['prompt']['info']['dl_cwind_data'])
        utils.set_format_custom_text(self.CWIND_CONFIG['data_name_length'])
        for year in self.years:
            for stn in self.year_station[year]:
                self._download_single_cwind_data(stn, year)

    def _station_in_a_year(self, year):
        """Get stations' id in specified year.

        """
        url = self.CWIND_CONFIG['urls']['data']
        page = requests.get(url)
        data = page.text
        soup = BeautifulSoup(data, features='lxml')
        stations = set()
        suffix = 'c%s.txt.gz' % year
        anchors = soup.find_all('a')

        for link in anchors:
            href = link.get('href')
            if href.endswith(suffix):
                stn_id = href[0:5]
                stations.add(stn_id)

        return stations

    def _download_single_station_info(self, station):
        """Download single cwind station information.

        """
        url = self.CWIND_CONFIG['urls']['stations']
        save_dir = self.CWIND_CONFIG['dirs']['stations']
        file_name = station + '.txt'
        file_path = save_dir + file_name

        if os.path.exists(file_path):
            return True

        payload = dict()
        payload['station'] = str.lower(station)
        try:
            html = requests.get(url, params=payload, verify=True)
        except Exception as msg:
            print(msg)
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
        self.cwind_station_file_names.append(file_name)
        print(station, end='\t', flush=True)
        return True

    def _download_single_cwind_data(self, station, year):
        """Download Continuous Wind data of specified station and year.

        """
        save_dir = self.CWIND_CONFIG['dirs']['data']
        data_url = self.CWIND_CONFIG['urls']['data']
        os.makedirs(save_dir, exist_ok=True)
        file_name = '{0}c{1}.txt.gz'.format(station, year)
        file_path = '{0}{1}'.format(save_dir, file_name)
        file_url = '{0}{1}'.format(data_url, file_name)
        if utils.download(file_url, file_path):
            self.cwind_data_file_names.append(file_name)
