import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy.orm import mapper
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
import matplotlib.pyplot as plt

import utils

Base = declarative_base()

class Regression(object):

    def __init__(self, CONFIG, period, region, passwd):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        utils.setup_database(self, Base)

        self.read_era5_smap()

    def _get_table_class_by_name(self, table_name):

        class Target(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Target, t)

            return Target

    def read_era5_smap(self):
        table_name = 'smap_2015'

        df = pd.read_sql('SELECT * FROM smap_2015', self.engine)
        df.sample(frac=1)

        df.hist(figsize = (12,10))
        plt.show()
