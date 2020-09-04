import logging

from sqlalchemy.ext.declarative import declarative_base

import utils

Base = declarative_base()

class TableCombiner(object):

    def __init__(self, CONFIG, period, region, basin, passwd):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.basin = basin

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]

        self.combine()

    def combine(self):
        # tablenames = []
        # for year in self.years:
        #     tablenames.append(f'tc_sfmr_era5_{year}_{self.basin}')

        # utils.combine_tables(self, 'tc_sfmr_era5_na', tablenames,
        #                      ['sfmr_datetime_lon_lat'])

        tablenames = []
        for year in self.years:
            tablenames.append(f'tc_smap_era5_{year}_{self.basin}')

        utils.combine_tables(self, 'tc_smap_era5_na', tablenames,
                             ['satel_datetime_lon_lat'])
