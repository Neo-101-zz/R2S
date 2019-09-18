import datetime
import logging
import math

import numpy as np
import netCDF4
from readNetCDF_CCMP import ReadNetcdf
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, MetaData
from sqlalchemy import Integer, Float, String, DateTime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns

import cwind
import utils

file = './CCMP_Wind_Analysis_19961006_V02.0_L3.0_RSS.nc'

Base = declarative_base()

class CompareCCMPWithInStu(object):

    def __init__(self, CONFIG, period, region, passwd):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.hours = {0: 0, 1: 6, 2: 12, 3: 18}
        self._setup_db()

        # After once running, this line is blocked
        # Run it when necessary
        # self._add_cwind_station_dis2coast()

        self._compare_with_cwind(file)

    def _setup_db(self):
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

    def _add_cwind_station_dis2coast(self):
        self.logger.info(('Adding column of distance to coast to table '
                          + 'of cwind station'))
        col_dis2coast = Column('distance_to_coast', Float())

        cwind_station_class = utils.get_class_by_tablename(
            self.engine, cwind.CwindStation.__tablename__)

        if not hasattr(cwind_station_class, col_dis2coast.name):
            utils.add_column(self.engine, cwind.CwindStation.__tablename__,
                             col_dis2coast)

        # Do NOT directly query cwind.CwindStation
        # Beacause due to some reason, its new column's value cannot
        # be added
        station_query = self.session.query(cwind_station_class)
        total = station_query.count()
        for idx, stn in enumerate(station_query):
            print(f'\r{stn.id} ({idx+1}/{total})', end='')
            stn.distance_to_coast = self._distance_from_coast(
                stn.latitude, stn.longitude)

        self.session.commit()

        utils.delete_last_lines()
        print()

    def _compare_with_cwind(self, ccmp_file_path):
        file = ccmp_file_path.split('/')[-1]
        base_datetime = datetime.datetime(
            year=int(file[19:23]), month=int(file[23:25]),
            day=int(file[25:27]), hour=0, minute=0, second=0)

        dis2coast_array = []
        wspd_absolute_error = []
        wdir_absolute_error = []

        vars = netCDF4.Dataset(ccmp_file_path).variables
        ccmp_lat = vars['latitude']
        ccmp_lon = vars['longitude']

        lat_padding = np.zeros(92)
        ccmp_lat = np.append(ccmp_lat, lat_padding, axis=0)
        ccmp_lat = np.roll(ccmp_lat, 46, axis=0)

        cwind_station_class = utils.get_class_by_tablename(
            self.engine, cwind.CwindStation.__tablename__)

        cwind_station_query = self.session.query(cwind_station_class)
        total = cwind_station_query.count()
        count = 0

        for stn in cwind_station_query:
            count += 1
            info = f'Comparing CCMP with cwind station {stn.id}'
            print(f'\r{info} ({count}/{total})', end='')
            # extract cwind speed and direction
            cwind_data_table_name = f'cwind_{stn.id}'
            CwindData = utils.get_class_by_tablename(
                self.engine, cwind_data_table_name)
            if CwindData is None:
                return None, None

            for h in self.hours:
                target_datetime = (base_datetime + datetime.timedelta(
                    hours=self.hours[h]))
                cwind_match = self.session.query(CwindData).\
                        filter_by(datetime=target_datetime).first()
                if cwind_match is None:
                    continue

                map_padding = np.zeros((92, 1440))

                uwnd = vars['uwnd'][h, :, :]
                vwnd = vars['vwnd'][h, :, :]

                uwnd = np.append(uwnd, map_padding, axis=0)
                vwnd = np.append(vwnd, map_padding, axis=0)
                uwnd = np.roll(uwnd, 46, axis=0)
                vwnd = np.roll(vwnd, 46, axis=0)

                ccmp_wspd, ccmp_wdir = self._ccmp_near_cwind(
                    stn, ccmp_lat, ccmp_lon, uwnd, vwnd)

                if ccmp_wspd is None or ccmp_wdir is None:
                    continue

                cwind_wspd = cwind_match.wspd_10
                cwind_wdir = cwind_match.wdir

                dis2coast_array.append(stn.distance_to_coast)
                wspd_absolute_error.append(abs(cwind_wspd - ccmp_wspd))
                wdir_absolute_error.append(abs(cwind_wdir - ccmp_wdir))

        utils.delete_last_lines()
        print('Done')
        print('MAE of wind speed: ' + str(
            sum(wspd_absolute_error)/len(wspd_absolute_error)))
        print('MAE of wind direction: ' + str(
            sum(wdir_absolute_error)/len(wdir_absolute_error)))

        dis2coast_array = np.array(dis2coast_array)
        wspd_absolute_error = np.array(wspd_absolute_error)
        wdir_absolute_error = np.array(wdir_absolute_error)

        plt.subplot(2, 1, 1)
        ax_1 = sns.regplot(x=dis2coast_array, y=wspd_absolute_error,
                           color='b')
        plt.xlabel('Distance to coast (km)')
        plt.ylabel('Wind speed absolute_error (m/s)')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        ax_2 = sns.regplot(x=dis2coast_array, y=wdir_absolute_error,
                           color='g')
        plt.xlabel('Distance to coast (km)')
        plt.ylabel('Wind speed absolute_error (m/s)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig((f'{self.CONFIG["result"]["dirs"]["fig"]}'
                     + f'ccmp_cwind_absolute_error_dis2coast.png'))
        plt.show()

    def _ccmp_near_cwind(self, stn, ccmp_lat, ccmp_lon, uwnd, vwnd):
        # get the CCMP cell which cwind station fall into
        stn_lat = stn.latitude
        stn_lon = stn.longitude
        lat_indices = utils.find_index([stn_lat, stn_lat], 'lat')
        lon_indices = utils.find_index([stn_lon, stn_lon], 'lon')

        if (abs(ccmp_lat[lat_indices[0]] - stn_lat) <\
            abs(ccmp_lat[lat_indices[1]] - stn_lat)):
            ccmp_lat_idx = lat_indices[0]
        else:
            ccmp_lat_idx = lat_indices[1]

        if (abs(ccmp_lon[lon_indices[0]] - stn_lon) <\
            abs(ccmp_lon[lon_indices[1]] - stn_lon)):
            ccmp_lon_idx = lon_indices[0]
        else:
            ccmp_lon_idx = lon_indices[1]

        if (abs(ccmp_lat[ccmp_lat_idx] - stn_lat) > 0.25
            or abs(ccmp_lon[ccmp_lon_idx] - stn_lon) > 0.25):
            self.logger.error('Fail getting WVC near cwind station.')

        # calculate WVC's speed and direction
        ccmp_u_wspd = uwnd[ccmp_lat_idx][ccmp_lon_idx]
        ccmp_v_wspd = vwnd[ccmp_lat_idx][ccmp_lon_idx]
        ccmp_wspd = math.sqrt(ccmp_u_wspd**2 + ccmp_v_wspd**2)
        # Convert CCMP's Wind Vector Azimuth to
        # NDBC's Meteorological Wind Direction
        ccmp_wdir = math.degrees(math.atan2(ccmp_u_wspd, ccmp_v_wspd))
        ccmp_wdir_converted = (ccmp_wdir + 180) % 360

        return ccmp_wspd, ccmp_wdir_converted

    def _distance_from_coast(self, lat, lon, resolution='l', degree_in_km=111.12):
        plt.ioff()

        m = Basemap(llcrnrlon=0.0, llcrnrlat=-90.0, urcrnrlon=360.0,
                    urcrnrlat=90.0, projection='cyl', resolution=resolution)
        coast = m.drawcoastlines()

        coordinates = np.vstack(coast.get_segments())
        lons,lats = m(coordinates[:,0],coordinates[:,1],inverse=True)

        dists = np.sqrt((lons-lon)**2+(lats-lat)**2)

        return float(np.min(dists)*degree_in_km)
