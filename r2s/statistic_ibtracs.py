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

        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name'][
            self.basin]
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)
        self.tc_query = self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        )
        self.tc_query_num = self.tc_query.count()

        self.detect_rapid_intensification()
        # self.how_fast_tcs_move()
        # self.simple_statistic_of_tcs_moving_speed()
        # self.how_fast_tcs_intensity_change()
        # self.simple_statistic_of_tc_intensity_change()

    def simple_statistic_of_tc_intensity_change(self):
        table_name = self.CONFIG['ibtracs']['table_name'][
            'statistic']['intensity_change'][self.basin]
        df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
        change = df['intensity_change_percent']
        change_speed = df['intensity_change_percent_per_hour']

        print((f"""Simple statistic of how LARGE TCs\' intensity """
               f"""change (%)"""))
        print(f'Count: {len(df)}')
        print(f'Median of absolute value: {change.abs().median()}')
        print(f'Mean of absoulte value: {change.abs().mean()}')
        print(f'Max: {change.max()}')
        print(f'Min: {change.min()}')
        print()

        print((f"""Simple statistic of how FAST TCs\' intensity """
               f"""change (%/h)"""))
        print(f'Count: {len(df)}')
        print(f'Median of absoulte value: {change_speed.abs().median()}')
        print(f'Mean of absoulte value: {change_speed.abs().mean()}')
        print(f'Max: {change_speed.max()}')
        print(f'Min: {change_speed.min()}')
        print()

    def detect_rapid_intensification(self):
        self.logger.info('Detecting rapid intensification')

        for i, tc in enumerate(self.tc_query):
            print(f'\r{i+1}/{self.tc_query_num}', end='')
            # find next TC
            if tc.wind is None or i == self.tc_query_num :
                continue

            for offset, next_tc in enumerate(self.tc_query[i+1:]):
                j = i+1+offset
                if next_tc.wind is None or j == self.tc_query_num:
                    continue

                delta = next_tc.date_time - tc.date_time

                if (not (delta.days == 1 and delta.seconds == 0)
                        or tc.sid != next_tc.sid):
                    continue

                # duration, shift = self.cal_before_speed(tc, next_tc)
                intensity_change, intensity_change_percent = \
                        self.cal_intensity_change(tc, next_tc)
                if intensity_change >= 30:
                    print((f"""{tc.name} {tc.date_time} - """
                           f"""{next_tc.date_time} {intensity_change}"""))
                # hours = duration / 60

    def how_fast_tcs_intensity_change(self):
        self.logger.info('Calculating how fast TCs\' intensity change')

        # create table for recording
        TCIntensityChange = self.create_tc_intensity_change_table()

        table_rows = []
        for idx, tc in enumerate(self.tc_query):
            print(f'\r{idx+1}/{self.tc_query_num}', end='')
            # find next TC
            if idx < self.tc_query_num and tc.wind is not None:
                next_idx = idx + 1
                while((next_idx < self.tc_query_num
                       and self.tc_query[next_idx].wind is None)):
                    next_idx += 1
                if next_idx == self.tc_query_num:
                    break

                next_tc = self.tc_query[next_idx]

                if tc.sid != next_tc.sid:
                    continue
            else:
                continue

            duration, shift = self.cal_before_speed(tc, next_tc)
            intensity_change, intensity_change_percent = \
                    self.cal_intensity_change(tc, next_tc)
            hours = duration / 60

            # record into table
            row = TCIntensityChange()
            row.sid = tc.sid
            row.name = tc.name
            row.basin = tc.basin
            row.start_datetime = tc.date_time
            row.duration_in_mins = duration
            row.shift_in_kms = shift
            row.intensity_change = intensity_change
            row.intensity_change_percent = intensity_change_percent
            row.intensity_change_per_hour = intensity_change / hours
            row.intensity_change_percent_per_hour = \
                    intensity_change_percent / hours
            row.sid_start_datetime = f'{tc.sid}_{tc.date_time}'

            table_rows.append(row)

        if len(table_rows):
            utils.bulk_insert_avoid_duplicate_unique(
                table_rows, self.CONFIG['database']\
                ['batch_size']['insert'],
                TCIntensityChange, ['sid_start_datetime'], self.session,
                check_self=True)

        utils.delete_last_lines()
        print('Done')

    def simple_statistic_of_tcs_moving_speed(self):
        table_name = self.CONFIG['ibtracs']['table_name'][
            'statistic']['moving_speed'][self.basin]
        df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
        speed = df['speed_kmph']

        print('Simple statistic of how fast TCs move (km/h)')
        print(f'Count: {len(df)}')
        print(f'Median: {speed.median()}')
        print(f'Mean: {speed.mean()}')
        print(f'Max: {speed.max()}')
        print(f'Min: {speed.min()}')
        print()

    def how_fast_tcs_move(self):
        self.logger.info('Calculating how fast TCs move')

        # create table for recording
        TCMovingSpeed = self.create_tc_moving_speed_table()

        table_rows = []
        for idx, tc in enumerate(self.tc_query):
            print(f'\r{idx+1}/{self.tc_query_num}', end='')
            # find next TC
            if idx < self.tc_query_num:
                next_tc = self.tc_query[idx + 1]
                if tc.sid != next_tc.sid:
                    continue
            else:
                break

            duration, shift = self.cal_before_speed(tc, next_tc)
            speed = shift / (duration / 60)

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

    def cal_intensity_change(self, tc, next_tc):
        delta = next_tc.wind - tc.wind
        delta_percent = delta / tc.wind

        return delta, delta_percent

    def cal_before_speed(self, tc, next_tc):
        delta = next_tc.date_time - tc.date_time
        duration = delta.days * 1440 + delta.seconds // 60

        start_pt = (tc.lat,
                    utils.longitude_converter(tc.lon, '360', '-180'))
        end_pt = (next_tc.lat,
                  utils.longitude_converter(next_tc.lon, '360', '-180'))
        shift = distance.distance(start_pt, end_pt).km

        return duration, shift

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

    def create_tc_intensity_change_table(self):

        table_name = self.CONFIG['ibtracs']['table_name'][
            'statistic']['intensity_change'][self.basin]

        class TCIntensityChange(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(TCIntensityChange, t)

            return TCIntensityChange

        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('sid', String(13), nullable=False))
        cols.append(Column('name', String(50)))
        cols.append(Column('basin', String(2), nullable=False))
        cols.append(Column('start_datetime', DateTime, nullable=False))
        cols.append(Column('duration_in_mins', Integer, nullable=False))
        cols.append(Column('shift_in_kms', Float, nullable=False))
        cols.append(Column('intensity_change', Float, nullable=False))
        cols.append(Column('intensity_change_percent', Float,
                           nullable=False))
        cols.append(Column('intensity_change_per_hour', Float,
                           nullable=False))
        cols.append(Column('intensity_change_percent_per_hour', Float,
                           nullable=False))

        cols.append(Column('sid_start_datetime', String(50),
                           nullable=False, unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(TCIntensityChange, t)

        self.session.commit()

        return TCIntensityChange
