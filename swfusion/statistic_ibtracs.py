import logging

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import mapper

import utils

Base = declarative_base()

class Statisticer(object):

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


    def how_fast_tcs_moving(self):
        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name'][
            self.basin]
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)
        tc_query = self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        )
        total = tc_query.count()

        # create table for recording
        TCMovingSpeed = self.create_tc_moving_speed_table()

        for idx, tc in enumerate(tc_query):
            # find next TC
            if idx != total - 1:
                next_tc = tc_query[idx + 1]
                if tc.sid != next_tc.sid:
                    continue
            else:
                break

            # calculate shift and speed
            shift, speed = self.cal_shift_speed(tc, next_tc)

            # record into table

    def create_tc_moving_speed_table(self):

        table_name = self.CONFIG['ibtracs']['table_name'][
            'statistic']['moving_speed'][self.basin]

        class TCMovingSpeed(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(TCMovingSpeed, t)

            return TCMovingSpeed

        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('sid', String(13), nullable=False))
        cols.append(Column('name', String(50)))
        cols.append(Column('basin', String(2), nullable=False))
        cols.append(Column('start_datetime', DateTime, nullable=False))
        cols.append(Column('duration_in_mins', Integer, nullable=False))
        cols.append(Column('shift_in_kms', Float, nullable=False))
        cols.append(Column('speed_kmpm', Float, nullable=False))

        cols.append(Column('sid_start_datetime', String(50),
                           nullable=False, unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(TCMovingSpeed, t)

        self.session.commit()

        return TCMovingSpeed
