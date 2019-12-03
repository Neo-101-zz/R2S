import datetime
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sqlalchemy.ext.declarative import declarative_base
import requests
from bs4 import BeautifulSoup
import mysql.connector
from sqlalchemy import create_engine, extract
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, Date, DateTime
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import mapper
import pandas as pd

import utils

Base = declarative_base()

class ISDManager(object):

    def __init__(self, CONFIG, period, region, passwd):
        self.logger = logging.getLogger(__name__)

        self.CONFIG = CONFIG
        self.period = period
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.zorders = {
            'coastlines': 4,
            'mapboundary': 0,
            'continents': 8,
            'station': 9,
            'contour': 3,
            'contourf': 2,
            'wedge': 7,
            'grid': 10
        }

        utils.setup_database(self, Base)

        # self.read_scs_stations()

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.if_station_on_land()
        self.plot_stations_on_map()

        self.download_and_read_scs_data()

    def plot_stations_on_map(self):
        stn_df = pd.read_sql('isd_scs_stations', self.engine)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        utils.draw_SCS_basemap(self, ax)

        geo_axs = ['lon', 'lat']
        stn_locs = ['open sea', 'onshore', 'offshore']
        stn_lonlat = dict()
        for axis in geo_axs:
            stn_lonlat[axis] = dict()
            for loc in stn_locs:
                stn_lonlat[axis][loc] = []

        stn_num = len(stn_df['station_id'])
        stn_idx_in_period = []
        for i in range(stn_num):
            begin = stn_df['begin_date'][i].date()
            end = stn_df['end_date'][i].date()
            if (begin > self.period[1].date()
                or end < self.period[0].date()):
                continue

            stn_idx_in_period.append(i)

            for axis in geo_axs:
                stn_lonlat[axis][stn_df['comment'][i]].append(
                    stn_df[axis][i])

        stns = dict()
        for loc in stn_locs:
            stns[loc] = (stn_lonlat['lon'][loc], stn_lonlat['lat'][loc])

        data = (stns['open sea'], stns['onshore'], stns['offshore'])
        colors = ('red', 'green', 'blue')
        groups = stn_locs

        for data, color, group in zip(data, colors, groups):
            x, y = data
            ax.scatter(x, y, alpha=0.8, c=color,
                                 edgecolors='none', s=30, label=group,
                                 zorder=self.zorders['station'])

        for i in stn_idx_in_period:
            text = ax.annotate(stn_df['name'][i],
                        (stn_df['lon'][i], stn_df['lat'][i]))
            text.set_fontsize(7)
            text.set_zorder(self.zorders['grid'] + 1)

        plt.legend().set_zorder(self.zorders['grid'] + 1)

        fig_dir = self.CONFIG['result']['dirs']['fig']
        fig_name = (f"""ISD_SCS_stations_{self.period[0]}"""
                    f"""_{self.period[1]}.png""")
        fig_path = f'{fig_dir}{fig_name}'
        plt.savefig(fig_path)

    def if_station_on_land(self):
        ISDStation = utils.get_class_by_tablename(self.engine,
                                                  'isd_scs_stations')
        Grid = utils.get_class_by_tablename(self.engine, 'grid')
        all_stn_in_sea = True

        for stn in self.session.query(ISDStation):
            y, x = utils.get_latlon_index_of_closest_grib_point(
                stn.lat, stn.lon, self.grid_lats, self.grid_lons)
            pt = self.session.query(Grid).filter(
                Grid.x == x, Grid.y == y).first()
            if pt.land:
                all_stn_in_sea = False
                print((f"""{stn.station_id} is on land, with comment: """
                       f"""{stn.comment} """
                       f"""\tstn_lon: {stn.lon}\tstn_lat: {stn.lat}"""
                       f"""\tgrid_lon: {pt.lon}\tgrid_lat: {pt.lat}"""))
        if all_stn_in_sea:
            print(f'All ISD stations are in sea.')

    def download_and_read_scs_data(self):
        self.logger.info(f'Downloading ISD data')
        ISDStation = self.create_isd_station_table()

        for year in self.years:
            year_dir = f"{self.CONFIG['isd']['dirs']['csvs']}{year}/"
            os.makedirs(year_dir, exist_ok=True)

            stn_query = self.session.query(ISDStation).filter(
                extract('year', ISDStation.begin_date) <= year,
                extract('year', ISDStation.end_date) >= year
            )
            total = stn_query.count()
            count = 0

            ISDWind = self.create_isd_wind_table(year)

            for stn in stn_query:
                count += 1
                print((f"""\rDownloading and reading {stn.station_id} """
                       f"""in {year} {count}/{total}"""), end='')
                csv_path = self.download_stn_data_in_a_year(
                    stn, year, year_dir)

                self.read_isd_csv(ISDWind, csv_path, year)

            utils.delete_last_lines()
            print(f'{year} done')

    def read_isd_csv(self, ISDWind, csv_path, year):
        df = pd.read_csv(csv_path)
        pts_to_insert = []

        row_num = len(df['STATION'])
        for i in range(row_num):
            try:
                dt = datetime.datetime.strptime(df['DATE'][i],
                                                '%Y-%m-%dT%H:%M:%S')
                if dt < self.period[0] or dt > self.period[1]:
                    continue
                pt = ISDWind()

                wind_parts = df['WND'][i].split(',')
                pt.winddir = int(wind_parts[0])
                pt.windspd = 0.1 * float(wind_parts[3])
                # Missing
                if pt.winddir == 999 or pt.windspd == 999.9:
                    continue
                pt.winddir_quality_code = wind_parts[1]
                pt.wind_type_code = wind_parts[2]
                pt.windspd_quality_code = wind_parts[4]

                pt.station_id = str(df['STATION'][i])
                pt.date_time = dt

                pt.lon = float(df['LONGITUDE'][i])
                pt.lat = float(df['LATITUDE'][i])
                pt.y, pt.x = utils.get_latlon_index_of_closest_grib_point(
                    pt.lat, pt.lon, self.grid_lats, self.grid_lons)
                pt.elevation = float(df['ELEVATION'][i])

                pt.station_id_datetime = (f"""{pt.station_id}"""
                                          f"""_{pt.date_time}""")

                pts_to_insert.append(pt)
            except Exception as msg:
                breakpoint()
                exit(msg)

        utils.bulk_insert_avoid_duplicate_unique(
            pts_to_insert, self.CONFIG['database']\
            ['batch_size']['insert'],
            ISDWind, ['station_id_datetime'], self.session,
            check_self=True)

    def create_isd_wind_table(self, year):
        table_name = f'isd_scs_{year}'

        class ISDWind(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(ISDWind, t)

            return ISDWind

        cols = []
        # IBTrACS columns
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('station_id', String(30), nullable=False))
        cols.append(Column('date_time', DateTime, nullable=False))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('elevation', Float, nullable=False))

        cols.append(Column('wind_type_code', String(1),
                           nullable=False))
        cols.append(Column('windspd', Float, nullable=False))
        cols.append(Column('winddir', Integer, nullable=False))
        cols.append(Column('windspd_quality_code', String(1),
                           nullable=False))
        cols.append(Column('winddir_quality_code', String(1),
                           nullable=False))

        cols.append(Column('station_id_datetime', String(60),
                           nullable=False, unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(ISDWind, t)

        self.session.commit()

        return ISDWind

    def download_stn_data_in_a_year(self, stn, year, year_dir):

        root_url = self.CONFIG['isd']['urls']['csv']
        year_url = f'{root_url}{year}/'

        """
        page = requests.get(year_url)
        data = page.text
        soup = BeautifulSoup(data, features='lxml')
        anchors = soup.find_all('a')

        for link in anchors:
            href = link.get('href')
            if stn.station_id in href and href.endswith('.csv'):
                file_url = f'{year_url}{href}'
                file_path = f'{year_dir}{href}'

                utils.download(file_url, file_path)
        """

        file_url = f'{year_url}{stn.station_id}.csv'
        file_path = f'{year_dir}{stn.station_id}.csv'
        utils.download(file_url, file_path)

        return file_path

    def read_scs_stations(self):
        ISDStation = self.create_isd_station_table()

        csv_dir = self.CONFIG['isd']['dirs']['stations']
        csv_names = [x for x in os.listdir(csv_dir)
                     if x.endswith('csv')]
        for csv_name in csv_names:
            csv_path = f'{csv_dir}{csv_name}'
            self.read_station_info_from_csv(ISDStation, csv_path)

    def read_station_info_from_csv(self, ISDStation, csv_path):
        stns_in_csv = []

        df = pd.read_csv(csv_path)
        row_num, col_num = df.shape

        for i in range(row_num):
            stn = ISDStation()

            stn.station_id = str(df['STATION_ID'][i])
            stn.name = df['STATION'][i]
            stn.begin_date = datetime.datetime.strptime(
                df['BEGIN_DATE'][i], '%Y-%m-%d').date()
            stn.end_date = datetime.datetime.strptime(
                df['END_DATE'][i], '%Y-%m-%d').date()
            stn.lon = float(df['LONGITUDE'][i])
            stn.lat = float(df['LATITUDE'][i])
            stn.elevation = float(df['ELEVATION_(M)'][i])
            state = df['STATE'][i]
            if str(state) != 'nan':
                stn.state = str(state)
            country = df['COUNTRY'][i]
            if str(country) != 'nan':
                stn.country = str(country)

            stns_in_csv.append(stn)

        utils.bulk_insert_avoid_duplicate_unique(
            stns_in_csv, self.CONFIG['database']\
            ['batch_size']['insert'],
            ISDStation, ['station_id'], self.session,
            check_self=True)

    def create_isd_station_table(self):
        table_name = 'isd_scs_stations'

        class ISDStation(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(ISDStation, t)

            return ISDStation

        cols = []
        # IBTrACS columns
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('station_id', String(30), nullable=False,
                           unique=True))
        cols.append(Column('name', String(70), nullable=False))
        cols.append(Column('begin_date', Date, nullable=False))
        cols.append(Column('end_date', Date, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('elevation', Float, nullable=False))
        cols.append(Column('state', String(50)))
        cols.append(Column('country', String(20)))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(ISDStation, t)

        self.session.commit()

        return ISDStation
