import datetime
import logging
import os

import cdsapi
from sqlalchemy.ext.declarative import declarative_base

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
        self.download()

    def download(self):
        self._get_target_datetime()
        majority_dir = self.CONFIG['era5']['dirs']['rea_pres_lvl_maj']
        minority_dir = self.CONFIG['era5']['dirs']['rea_pres_lvl_min']
        os.makedirs(majority_dir, exist_ok=True)
        os.makedirs(minority_dir, exist_ok=True)

        c = cdsapi.Client()

        for year in self.dt_majority.keys():
            for month in self.dt_majority[year].keys():
                file_name = f'{majority_dir}{year}{str(month).zfill(2)}.grib'
                c.retrieve(
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
                    file_name)

        for year in self.dt_minority.keys():
            for month in self.dt_minority[year].keys():
                for day_str in self.dt_minority[year][month].keys():
                    file_name =\
                            (f'{minority_dir}{year}{str(month).zfill(2)}'
                             + f'{day_str}.grib')
                    c.retrieve(
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
                        file_name)

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
