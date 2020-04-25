import logging

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import mapper
from geopy import distance
import pandas as pd

import utils

Base = declarative_base()

class Statisticer(object):

    def __init__(self, CONFIG, period, basin, passwd):
        self.CONFIG = CONFIG
        self.period = period
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.basin = basin

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        # self.how_fast_tcs_move()
        self.simple_statistic_of_tcs_moving_speed()

    def simple_statistic_of_tcs_moving_speed(self):
        table_name = self.CONFIG['ibtracs']['table_name'][
            'statistic']['moving_speed'][self.basin]
        df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)

        print('Simple statistic of how fast TCs move')
        print(f'Count: {len(df)}')
        print(f'Median: {df["speed_kmph"].median()}')
        print(f'Mean: {df["speed_kmph"].mean()}')
        print(f'Max: {df["speed_kmph"].max()}')
        print(f'Min: {df["speed_kmph"].min()}')

    def how_fast_tcs_move(self):
        self.logger.info('Calculating how fast TCs move')
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

        table_rows = []
        for idx, tc in enumerate(tc_query):
            print(f'\r{idx+1}/{total}', end='')
            # find next TC
            if idx != total - 1:
                next_tc = tc_query[idx + 1]
                if tc.sid != next_tc.sid:
                    continue
            else:
                break

            # calculate shift and speed
            duration, shift, speed = self.cal_about_speed(tc, next_tc)

            # record into table
            row = TCMovingSpeed()
            row.sid = tc.sid
            row.name = tc.name
            row.basin = tc.basin
            row.start_datetime = tc.date_time
            row.duration_in_mins = duration
            row.shift_in_kms = shift
            row.speed_kmph = speed
            row.sid_start_datetime = f'{tc.sid}_{tc.date_time}'

            table_rows.append(row)

        if len(table_rows):
            utils.bulk_insert_avoid_duplicate_unique(
                table_rows, self.CONFIG['database']\
                ['batch_size']['insert'],
                TCMovingSpeed, ['sid_start_datetime'], self.session,
                check_self=True)

        utils.delete_last_lines()
        print('Done')

    def cal_about_speed(self, tc, next_tc):
        delta = next_tc.date_time - tc.date_time
        duration = delta.days * 1440 + delta.seconds // 60

        start_pt = (tc.lat,
                    utils.longtitude_converter(tc.lon, '360', '-180'))
        end_pt = (next_tc.lat,
                  utils.longtitude_converter(next_tc.lon, '360', '-180'))
        shift = distance.distance(start_pt, end_pt).km

        speed = shift / (duration / 60)

        return duration, shift, speed

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
        cols.append(Column('speed_kmph', Float, nullable=False))

        cols.append(Column('sid_start_datetime', String(50),
                           nullable=False, unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(TCMovingSpeed, t)

        self.session.commit()

        return TCMovingSpeed
