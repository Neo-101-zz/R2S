import datetime
import logging
import os

import cdsapi
import pygrib
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime
from sqlalchemy import Column

import ibtracs
import utils

Base = declarative_base()

class ERA5Manager(object):
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
        self.main_hours = [0, 6, 12, 18]
        self.lat_grid_p = [y*0.25-90 for y in range(721)]
        self.lon_grid_p = [x*0.25-90 for x in range(1440)]

        self.cdsapi_client = cdsapi.Client()
        self.vars = [
            'geopotential','relative_humidity','temperature',
            'u_component_of_wind','v_component_of_wind'
        ]
        self.pres_lvl = [
            '100','200','300',
            '400','500','600',
            '700','800','850',
            '925','975','1000'
        ]
        utils.setup_database(self, Base)
        self.download_and_read()

    def download_majority(self, file_path, year, month):
        self.cdsapi_client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type':'reanalysis',
                'format':'grib',
                'variable':self.vars,
                'pressure_level':self.pres_lvl,
                'year':f'{year}',
                'month':str(month).zfill(2),
                'day':list(self.dt_majority[year][month]['day']),
                'time':list(self.dt_majority[year][month]['time'])
            },
            file_path)

    def download_minority(self, file_path, year, month, day_str):
        self.cdsapi_client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type':'reanalysis',
                'format':'grib',
                'variable':self.vars,
                'pressure_level':self.pres_lvl,
                'year':f'{year}',
                'month':str(month).zfill(2),
                'day':day_str,
                'time':list(self.dt_minority[year][month][day_str])
            },
            file_path)

    def download_and_read(self):
        self.logger.info('Downloading and reading ERA5 reanalysis')
        self._get_target_datetime()
        majority_dir = self.CONFIG['era5']['dirs']['rea_pres_lvl_maj']
        minority_dir = self.CONFIG['era5']['dirs']['rea_pres_lvl_min']
        os.makedirs(majority_dir, exist_ok=True)
        os.makedirs(minority_dir, exist_ok=True)

        for year in self.dt_majority.keys():
            for month in self.dt_majority[year].keys():
                file_path = f'{majority_dir}{year}{str(month).zfill(2)}.grib'
                self.logger.info(f'Downloading majority {file_path}')
                if not os.path.exists(file_path):
                    self.download_majority(file_path, year, month)
                self.logger.info(f'Reading majority {file_path}')
                self.read(file_path)
                os.remove(file_path)

        for year in self.dt_minority.keys():
            for month in self.dt_minority[year].keys():
                for day_str in self.dt_minority[year][month].keys():
                    file_path =\
                            (f'{minority_dir}{year}{str(month).zfill(2)}'
                             + f'{day_str}.grib')
                    self.logger.info(f'Downloading minority {file_path}')
                    if not os.path.exists(file_path):
                        self.download_minority(file_path, year, month, day_str)
                    self.logger.info(f'Reading minority {file_path}')
                    self.read(file_path)
                    os.remove(file_path)

    def read(self, file_path):
        # load grib file
        grbs = pygrib.open(file_path)
        # Alter TC table
        tc_table_name = ibtracs.WMOWPTC.__tablename__
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        has_add = set()
        for m in range(grbs.messages):
            grb = grbs.message(m+1)
            message_name = grb.name.replace(' ', '_').lower()
            col_name = f'{message_name}_{grb.level}'
            if col_name not in has_add and not hasattr(TCTable, col_name):
                breakpoint()
                mysql_connector = utils.get_mysql_connector(self)
                col = Column(col_name, Float())
                self.logger.info((f'Adding column {col_name} '
                                  + f'to table {tc_table_name}'))
                utils.add_column(mysql_connector,
                                 self.engine, tc_table_name, col)
                has_add.add(col_name)
                mysql_connector.close()
        # Update TC table
        self.session.commit()
        # loop TC table
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        tc_query = self.session.query(TCTable)
        total = tc_query.count()
        count = 0
        info = f'Reading reanalysis data of TC records'
        self.logger.info(info)
        # get lat and lon of row
        for row in tc_query:
            # Get range of matching cell
            lat1, lat2, lon1, lon2 = self._get_subset_range_of_grib(
                row.lat, row.lon)
            tc_datetime = row.datetime
            count += 1
            print(f'\r{info} {count}/{total}', end='')

            # read out variables
            for m in range(grbs.messages):
                grb = grbs.message(m+1)
                grb_date, grb_time = str(grb.dataDate), str(grb.dataTime)
                if grb_time == '0':
                    grb_time = '000'
                grb_datetime = datetime.datetime.strptime(
                    f'{grb_date}{grb_time}', '%Y%m%d%H%M%S')
                if tc_datetime != grb_datetime:
                    continue

                # extract corresponding cell in ERA5 reanalysis file
                data, lats, lons = grb.data(lat1, lat2, lon1, lon2)
                data = utils.convert_dtype(data)
                if data == grb.missingValue:
                    continue
                name = f'{grb.name.replace(" ", "_").lower()}_{grb.level}'
                # Add column value
                setattr(row, name, data)
        # write into DB
        self.session.commit()
        utils.delete_last_lines()
        print('Done')

    def _get_subset_range_of_grib(self, lat, lon):
        lat_ae = [abs(lat-y) for y in self.lat_grid_p]
        lon_ae = [abs(lon-x) for x in self.lon_grid_p]

        lat_match = self.lat_grid_p[lat_ae.index(min(lat_ae))]
        lon_match = self.lon_grid_p[lon_ae.index(min(lon_ae))]

        lat1 = lat_match if lat > lat_match else lat
        lat2 = lat_match if lat < lat_match else lat
        lon1 = lon_match if lon > lon_match else lon
        lon2 = lon_match if lon < lon_match else lon

        return lat1, lat2, lon1, lon2

    def _update_majority_datetime_dict(self, dt_dict, year, month, day, hour):
        day_str = str(day).zfill(2)
        time_str = f'{str(hour).zfill(2)}:00'

        if year not in dt_dict:
            dt_dict[year] = dict()
        if month not in dt_dict[year]:
            dt_dict[year][month] = dict()
            dt_dict[year][month]['day'] = set()
            dt_dict[year][month]['time'] = set()

        dt_dict[year][month]['day'].add(day_str)
        dt_dict[year][month]['time'].add(time_str)

    def _update_minority_datetime_dict(self, dt_dict, year, month, day, hour):
        day_str = str(day).zfill(2)
        time_str = f'{str(hour).zfill(2)}:00'

        if year not in dt_dict:
            dt_dict[year] = dict()
        if month not in dt_dict[year]:
            dt_dict[year][month] = dict()
        if day_str not in dt_dict[year][month]:
            dt_dict[year][month][day_str] = set()

        dt_dict[year][month][day_str].add(time_str)

    def _get_target_datetime(self):
        tc_query = self.session.query(ibtracs.WMOWPTC)

        dt_majority = dict()
        dt_minority = dict()

        for row in tc_query:
            year, month = row.datetime.year, row.datetime.month
            day, hour = row.datetime.day, row.datetime.hour
            if hour in self.main_hours:
                self._update_majority_datetime_dict(dt_majority, year,
                                                    month, day, hour)
            else:
                self._update_minority_datetime_dict(dt_minority, year,
                                                    month, day, hour)

        self.dt_majority = dt_majority
        self.dt_minority = dt_minority
