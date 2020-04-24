import datetime
import logging
import time

import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, Date
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import mapper
from global_land_mask import globe

import isd
import era5
import satel_scs
import utils

Base = declarative_base()

class StatisticManager(object):

    def __init__(self, CONFIG, period, region, passwd, save_disk):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.save_disk = save_disk
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        utils.setup_database(self, Base)

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.sources = ['era5', 'smap']

        # self.compare_with_isd()
        self.visualization()

    def compare_with_isd(self):
        """Compare wind speed from different data sources with
        ISD's wind speed.

        """
        # Get ISD windspd
        isd_manager = isd.ISDManager(self.CONFIG, self.period,
                                     self.region, self.db_root_passwd,
                                     work_mode='')
        # Download ISD csvs in period
        isd_csv_paths = isd_manager.download_and_read_scs_data()

        # Get windspd from different sources

        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name']['scs']
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)

        sources_str = ''
        for idx, src in enumerate(self.sources):
            if idx < len(self.sources) - 1:
                sources_str = f"""{sources_str}{src.upper()} and """
            else:
                sources_str = f"""{sources_str}{src.upper()}"""

        # Filter TCs during period
        for tc in self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        ).yield_per(self.CONFIG['database']['batch_size']['query']):
            # 
            if tc.wind < 64:
                continue
            if tc.r34_ne is None:
                continue
            if bool(globe.is_land(tc.lat, tc.lon)):
                continue
            # Draw windspd from CCMP, ERA5, Interium
            # and several satellites
            # self.just_download_era5_equivalent_wind(tc)
            success = self.get_concurrent_data(isd_csv_paths, tc)
            if success:
                print((f"""Comparing {sources_str} with ISD record """
                       f"""when TC {tc.name} existed on """
                       f"""{tc.date_time}"""))
            else:
                print((f"""Skiping comparsion of {sources_str} with """
                       f"""ISD record when TC {tc.name} existed """
                       f"""on {tc.date_time}"""))
        print('Done')

    def get_concurrent_data(self, isd_csv_paths, tc):
        ISDBasedComparsion = self.create_isd_based_comparsion_table()
        comparsion_when_tc = []

        for csv_path in isd_csv_paths[tc.date_time.year]:
            skip_station = False
            # st = time.time()
            row = self.get_comparsion_row(csv_path, ISDBasedComparsion,
                                          tc)
            # print(f'isd: {time.time() - st}')
            if row is None:
                continue
            for src in self.sources:
                # st = time.time()
                row = self.update_comparsion_row(src, row)
                # print(f'{src}: {time.time() - st}')
                if row is None:
                    skip_station = True
                    break
            if skip_station:
                continue

            comparsion_when_tc.append(row)

        if not len(comparsion_when_tc):
            return False

        # Insert
        utils.bulk_insert_avoid_duplicate_unique(
            comparsion_when_tc, self.CONFIG['database']\
            ['batch_size']['insert'],
            ISDBasedComparsion , ['station_id_datetime'], self.session,
            check_self=True)

        return True

    def update_comparsion_row(self, src, row):
        if src == 'era5':
            return self.update_comparsion_row_with_era5(row)
        elif src == 'smap':
            return self.update_comparsion_row_with_smap(row)

    def just_download_era5_equivalent_wind(self, tc):
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = \
                era5_manager.download_surface_vars_of_whole_day(
                    'single_levels', 'surface_wind', tc.date_time)

    def update_comparsion_row_with_era5(self, row):
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = \
                era5_manager.download_surface_vars_of_whole_day(
                    'single_levels', 'surface_wind', row.date_time)
        windspd = utils.get_pixel_of_era5_windspd(
            era5_file_path, 'single_levels', row.date_time,
            row.lon, row.lat)

        row.era5_windspd = windspd

        return row

    def update_comparsion_row_with_smap(self, row):
        satel_manager = satel_scs.SCSSatelManager(
            self.CONFIG, self.period, self.region, self.db_root_passwd,
            save_disk=self.save_disk, work=False)
        smap_file_path = satel_manager.download('smap', row.date_time)

        windspd = utils.get_pixel_of_smap_windspd(
            smap_file_path, row.date_time, row.lon, row.lat)

        if windspd is None:
            row.smap_windspd = -1.0
        else:
            row.smap_windspd = windspd

        return row

    def same_datetime_exists_in_col(self, col, strftime_format,
                                    target_dt):
        col = col.tolist()
        target_dt_str = target_dt.strftime(strftime_format)

        try:
            match_index = col.index(target_dt_str)
        except ValueError as msg:
            return False, None

        return True, match_index

    def get_comparsion_row(self, csv_path, ISDBasedComparsion, tc):
        # Open csv
        try:
            df = pd.read_csv(csv_path)
        except Exception as msg:
            breakpoint()
            exit(msg)

        found, match_index = self.same_datetime_exists_in_col(
            df['DATE'], '%Y-%m-%dT%H:%M:%S', tc.date_time)

        if not found:
            return None
        else:
            i = match_index
            # Write ISD data into comparsion row
            pt = ISDBasedComparsion()

            wind_parts = df['WND'][i].split(',')
            # Missing
            if int(wind_parts[3]) == 9999:
                return None
            pt.windspd = 0.1 * float(wind_parts[3])
            pt.wind_type_code = wind_parts[2]
            # Quality control of windspd
            if int(wind_parts[4]) != 1:
                return None
            pt.windspd_quality_code = wind_parts[4]

            pt.station_id = str(df['STATION'][i])
            pt.date_time = tc.date_time

            pt.lon = float(df['LONGITUDE'][i])
            pt.lat = float(df['LATITUDE'][i])
            pt.y, pt.x = \
                    utils.get_latlon_index_of_closest_grib_point(
                        pt.lat, pt.lon, self.grid_lats,
                        self.grid_lons)
            pt.elevation = float(df['ELEVATION'][i])
            pt.windspd = utils.convert_10(pt.windspd, pt.elevation)

            pt.station_id_datetime = (f"""{pt.station_id}"""
                                      f"""_{pt.date_time}""")
            return pt

    def visualization(self):
        table_name = self.CONFIG['statistic']['table_name']
        df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)

        rmse = ((df.windspd - df.era5_windspd) ** 2).mean() ** .5

        print((f"""RMSE between ISD and ERA5: {rmse}"""))

    def create_isd_based_comparsion_table(self):
        table_name = self.CONFIG['statistic']['table_name']

        class ISDBasedComparsion(object):
            pass

        # Return TC table if it exists
        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(ISDBasedComparsion, t)

            return ISDBasedComparsion

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
        cols.append(Column('windspd_quality_code', String(1),
                           nullable=False))

        cols.append(Column('station_id_datetime', String(60),
                           nullable=False, unique=True))

        for src in self.sources:
            cols += self.get_source_comparsion_cols(src)

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(ISDBasedComparsion, t)

        self.session.commit()

        return ISDBasedComparsion

    def get_source_comparsion_cols(self, src):
        if src == 'era5':
            return self.get_era5_comparsion_cols()
        elif src == 'smap':
            return self.get_smap_comparsion_cols()

    def get_era5_comparsion_cols(self):
        cols = []
        cols.append(Column('era5_windspd', Float, nullable=False))
        return cols

    def get_smap_comparsion_cols(self):
        cols = []
        cols.append(Column('smap_windspd', Float, nullable=False))
        return cols
