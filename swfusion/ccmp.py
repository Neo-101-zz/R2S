import datetime
import logging
import os
import pickle

from netCDF4 import Dataset
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, Date
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import mapper
from sqlalchemy import create_engine, extract
from global_land_mask import globe
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils

Base = declarative_base()

class CCMPManager(object):

    def __init__(self, CONFIG, period, region, passwd, work_mode):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.spa_resolu = dict()

        self.spa_resolu['ccmp'] = self.CONFIG['ccmp']\
                ['spatial_resolution']
        self.spa_resolu['grid'] = self.CONFIG['grid']\
                ['spatial_resolution']

        self.grid_pts = dict()

        self.grid_pts['ccmp'] = dict()
        self.grid_pts['ccmp']['lat'] = [
            y * self.spa_resolu['ccmp'] - 78.375 for y in range(
                self.CONFIG['ccmp']['lat_grid_points_number'])
        ]
        self.grid_pts['ccmp']['lon'] = [
            x * self.spa_resolu['ccmp'] + 0.125 for x in range(
                self.CONFIG['ccmp']['lon_grid_points_number'])
        ]

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.zorders = self.CONFIG['plot']['zorders']['scs_basemap']
        self._get_region_corners_indices()

        self.root_url = self.CONFIG['ccmp']['url']
        self.filename_prefix = self.CONFIG['ccmp']['filename']\
                ['prefix']
        self.filename_suffix = self.CONFIG['ccmp']['filename']\
                ['suffix']
        self.root_dir = self.CONFIG['ccmp']['dir']

        utils.set_format_custom_text(
            self.CONFIG['ccmp']['filename_length'])

        if work_mode == 'fetch':
            self.download('tc')
            self.read()
        elif work_mode == 'compare':
            self.compare_ccmp_with_ibtracs()
        elif work_mode == 'fetch_and_compare':
            self.download('tc')
            self.read()
            self.compare_ccmp_with_ibtracs()

    def _get_region_corners_indices(self):
        self.lat1_index = self.grid_pts['ccmp']['lat'].index(
            self.lat1 + 0.5 * self.spa_resolu['ccmp'])
        self.lat2_index = self.grid_pts['ccmp']['lat'].index(
            self.lat2 - 0.5 * self.spa_resolu['ccmp'])
        self.lon1_index = self.grid_pts['ccmp']['lon'].index(
            self.lon1 + 0.5 * self.spa_resolu['ccmp'])
        self.lon2_index = self.grid_pts['ccmp']['lon'].index(
            self.lon2 - 0.5 * self.spa_resolu['ccmp'])

    def get_latlon_and_match_index(self, lat_or_lon, latlon_idx):
        if lat_or_lon == 'lat':
            lat_of_row = self.grid_pts['ccmp']['lat']\
                    [self.lat1_index + latlon_idx]
            # Choose south grid point nearest to RSS point
            lat_match_index = self.grid_lats.index(
                    lat_of_row - 0.5 * self.spa_resolu['grid'])

            return lat_of_row, lat_match_index

        elif lat_or_lon == 'lon':
            lon_of_pt = self.grid_pts['ccmp']['lon']\
                    [self.lon1_index + latlon_idx]
            # Choose east grid point nearest to RSS point
            lon_match_index = self.grid_lons.index(
                    lon_of_pt + 0.5 * self.spa_resolu['grid'])

            return lon_of_pt, lon_match_index

    def download(self, mode):
        self.files_path = []

        if mode == 'all':
            self.download_all()
        elif mode == 'tc':
            self.download_tc()

    def download_ccmp_on_one_day(self, dt_cursor):
        utils.setup_signal_handler()
        filename = (f"""{self.filename_prefix}"""
                     f"""{dt_cursor.strftime('%Y%m%d')}"""
                     f"""{self.filename_suffix}""")
        url_prefix = (f"""{self.root_url}/Y{dt_cursor.year}/"""
                      f"""M{str(dt_cursor.month).zfill(2)}/""")
        file_dir = (f"""{self.root_dir}Y{dt_cursor.year}/"""
                    f"""M{str(dt_cursor.month).zfill(2)}/""")
        file_path = f'{file_dir}{filename}'
        file_url = f'{url_prefix}{filename}'

        if os.path.exists(file_path):
            return file_path

        self.logger.info((f"""Downloading {filename}"""))

        os.makedirs(file_dir, exist_ok=True)
        utils.download(file_url, file_path, progress=True)
        utils.reset_signal_handler()

        return file_path

    def download_all(self):
        self.logger.info(f'Downloading all CCMP files during period')

        delta = self.period[1] - self.period[0]
        for i in range(delta.days):
            dt_cursor = self.period[0] + datetime.timedelta(days=i)
            file_path = self.download_ccmp_on_one_day(dt_cursor)

            self.files_path.append(file_path)

    def download_tc(self):
        self.logger.info((f"""Downloading CCMP files which containing """
                          f"""TCs during period"""))

        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name']['scs']
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)
        downloaded_dates = set()
        # Filter TCs during period
        for tc in self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        ).yield_per(self.CONFIG['database']['batch_size']['query']):
            if tc.r34_ne is None:
                continue
            # Download corresponding CCMP files
            dt_cursor = tc.date_time
            if dt_cursor.date() in downloaded_dates:
                continue

            file_path = self.download_ccmp_on_one_day(dt_cursor)
            downloaded_dates.add(dt_cursor.date())

            self.files_path.append(file_path)

    def create_scs_ccmp_table(self, date_):
        table_name = (f"""ccmp_scs_{date_.year}_"""
                      f"""{str(date_.month).zfill(2)}"""
                      f"""{str(date_.day).zfill(2)}""")

        class CCMP(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(CCMP, t)

            return CCMP

        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('date_time', DateTime, nullable=False))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('datetime_x_y', String(40), nullable=False,
                           unique=True))

        cols.append(Column('nobs', Integer, nullable=False))
        cols.append(Column('u_wind', Float, nullable=False))
        cols.append(Column('v_wind', Float, nullable=False))
        cols.append(Column('windspd', Float, nullable=False))
        cols.append(Column('winddir', Float, nullable=False))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(CCMP, t)

        self.session.commit()

        return CCMP

    def read(self):
        utils.reset_signal_handler()
        self.logger.info((f"""Reading CCMP files"""))

        # Traverse file path
        for file_path in self.files_path:
            date_str = file_path.split('_')[3]
            vars = Dataset(file_path).variables
            date_ = datetime.datetime.strptime(date_str, '%Y%m%d').date()
            CCMP = self.create_scs_ccmp_table(date_)
            info = f"""Reading {file_path.split('/')[-1]}"""
            # Traverse 4 time in one day
            for hour_idx, hour in enumerate(range(0, 24, 6)):
                print(f"""\r{info} on {str(hour).zfill(2)}:00""", end='')
                one_hour_scs_ccmp = []
                time = datetime.time(hour, 0, 0)
                dt = datetime.datetime.combine(date_, time)

                subset = dict()
                var_names = ['nobs', 'uwnd', 'vwnd']

                for var_name in var_names:
                    subset[var_name] = vars[var_name][hour_idx][
                        self.lat1_index: self.lat2_index+1,
                        self.lon1_index: self.lon2_index+1
                    ]

                one_hour_scs_ccmp = self.get_ccmp_of_one_hour(
                    dt, CCMP, subset, var_names)

                # Insert into table
                utils.bulk_insert_avoid_duplicate_unique(
                    one_hour_scs_ccmp, self.CONFIG['database']\
                    ['batch_size']['insert'],
                    CCMP, ['datetime_x_y'], self.session,
                    check_self=True)
            utils.delete_last_lines()
            print(f"""{info}: Done""")

    def get_ccmp_of_one_hour(self, dt, CCMP, subset, var_names):
        ccmp_pts = []
        lats_num, lons_num = subset['uwnd'].shape

        for y in range(lats_num):

            lat_of_row, lat_match_index = \
                    self.get_latlon_and_match_index('lat', y)
            grid_pt_lat = self.grid_lats[lat_match_index]

            for x in range(lons_num):

                lon_of_pt, lon_match_index = \
                        self.get_latlon_and_match_index('lon', x)
                grid_pt_lon = self.grid_lons[lon_match_index]

                # if (bool(globe.is_land(lat_of_row, lon_of_pt))
                #     or bool(globe.is_land(grid_pt_lat, grid_pt_lon))):
                #     continue

                row = CCMP()

                row.date_time = dt
                row.x = int(self.grid_x[lon_match_index])
                row.y = int(self.grid_y[lat_match_index])

                row.lon = lon_of_pt
                row.lat = lat_of_row

                row.datetime_x_y = f'{row.date_time}_{row.x}_{row.y}'

                row.nobs = int(subset['nobs'][y][x])
                row.u_wind = float(subset['uwnd'][y][x])
                row.v_wind = float(subset['vwnd'][y][x])
                # Wait to be updated when adding ERA5 data
                row.windspd, row.winddir = utils.compose_wind(
                    row.u_wind, row.v_wind, 'o')

                ccmp_pts.append(row)

        return ccmp_pts

    def compare_ccmp_with_ibtracs(self):
        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name']['scs']
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)
        # Filter TCs during period
        for tc in self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        ).yield_per(self.CONFIG['database']['batch_size']['query']):
            # 
            if tc.r34_ne is None:
                continue
            self.compare_ccmp_with_one_tc_record(tc)

    def compare_ccmp_with_one_tc_record(self, tc):
        self.logger.info((f"""Comparing windspd of CCMP with IBTrACS """
                          f"""at {tc.date_time}"""))

        # Download corresponding CCMP files
        dt = tc.date_time
        CCMP = self.create_scs_ccmp_table(dt.date())
        lons, lats, windspd = self.get_xyz_of_ccmp_windspd(CCMP, dt)

        fig, axs = plt.subplots(1, 1, figsize=(15, 10), sharey=True)
        ax = axs

        utils.draw_ccmp_windspd(self, fig, ax, dt, lons, lats, windspd)
        radii_area = utils.draw_ibtracs_radii(ax, tc, self.zorders)

        dt_str = dt.strftime('%Y_%m%d_%H%M')
        fig_dir = self.CONFIG['result']['dirs']['fig']['ibtracs_vs_ccmp']
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'ccmp_vs_ibtracs_{dt_str}_{tc.name}.png'
        plt.savefig(f'{fig_dir}{fig_name}')
        plt.clf()

    def get_xyz_of_ccmp_windspd(self, CCMP, dt):
        ccmp = dict()
        ccmp['lon'] = []
        ccmp['lat'] = []
        ccmp['windspd'] = []

        grid_lons_lats = dict()

        for name in ['lons', 'lats']:
            pickle_path = self.CONFIG['grid']['pickle'][name]
            with open(pickle_path, 'rb') as f:
                grid_lons_lats[name] = pickle.load(f)

        query_for_count = self.session.query(CCMP).filter(
            extract('hour', CCMP.date_time) == dt.hour)
        total = query_for_count.count()
        del query_for_count

        if not total:
            return [], [], 0

        min_lon, max_lon = 999, -999
        min_lat, max_lat = 999, -999

        for row in self.session.query(CCMP).filter(
            extract('hour', CCMP.date_time) == dt.hour).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            lon = grid_lons_lats['lons'][row.x]
            ccmp['lon'].append(lon)
            if lon < min_lon:
                min_lon = lon
            if lon > max_lon:
                max_lon = lon

            lat = grid_lons_lats['lats'][row.y]
            ccmp['lat'].append(lat)
            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat

            ccmp['windspd'].append(row.windspd)

        if min_lon > max_lon or min_lat > max_lat:
            return [], [], 0

        grid_spa_resolu = self.CONFIG['grid']['spatial_resolution']
        # DO NOT use np.linspace, because the round error is larger than
        # 0.01
        lons = list(np.arange(min_lon, max_lon + 0.5 * grid_spa_resolu,
                              grid_spa_resolu))
        lats = list(np.arange(min_lat, max_lat + 0.5 * grid_spa_resolu,
                              grid_spa_resolu))
        lons = [round(x, 2) for x in lons]
        lats = [round(y, 2) for y in lats]

        windspd = np.zeros(shape=(len(lats), len(lons)),
                           dtype=float)

        for i in range(total):
            try:
                lon_idx = lons.index(ccmp['lon'][i])
                lat_idx = lats.index(ccmp['lat'][i])
            except Exception as msg:
                breakpoint()
                exit(msg)

            # Only for display wind cell according to satellite's
            # spatial resolution
            for y_offset in range(-2, 3):
                sub_lat_idx = lat_idx + y_offset
                if sub_lat_idx < 0 or sub_lat_idx >= len(lats):
                    continue

                for x_offset in range(-2, 3):
                    sub_lon_idx = lon_idx + x_offset
                    if sub_lon_idx < 0 or sub_lon_idx >= len(lons):
                        continue

                    windspd[sub_lat_idx][sub_lon_idx] = \
                            ccmp['windspd'][i]

        return lons, lats, windspd
