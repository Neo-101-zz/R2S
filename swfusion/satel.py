"""Manage downloading and reading ASCAT, QucikSCAT and Windsat data.

"""
import datetime
import math
import logging
import pickle
import os
import time
import sys

from geopy import distance
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import mapper
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

from ascat_daily import ASCATDaily
from quikscat_daily_v4 import QuikScatDaily
from windsat_daily_v7 import WindSatDaily
import utils
import cwind
import sfmr

Base = declarative_base()

class SatelManager(object):

    def __init__(self, CONFIG, period, region, passwd,
                 spatial_window, temporal_window):
        self.logger = logging.getLogger(__name__)
        self.satel_names = ['ascat', 'qscat', 'wsat']
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.spatial_window = spatial_window
        self.temporal_window = temporal_window

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]

        # self.download()
        self.read(read_all=True)
        # self.show()
        self.match()

    def download(self):
        utils.setup_signal_handler()
        # Check and update period for each satellite
        self.periods = dict()
        self.downloaded_file_path = dict()
        for satel_name in self.satel_names:
            correct, period = utils.check_and_update_period(
                self.period, self.CONFIG[satel_name]['period_limit'],
                self.CONFIG['workflow']['prompt'])
            if not correct:
                return
            self.periods[satel_name] = period
            config4one = self.CONFIG[satel_name]
            self.downloaded_file_path[satel_name] = []
            self._download_single_satel(config4one, satel_name, period)

    def read(self, read_all):
        """Read data into MySQL database.

        """
        utils.reset_signal_handler()

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

        # self._insert_satel(read_all)

    def show(self):
        for satel_name in self.satel_names:
            for year in self.years:
                satel_table_name = f'{satel_name}_{year}'

                class SatelTable(object):
                    pass

                if self.engine.dialect.has_table(self.engine,
                                                 satel_table_name):
                    metadata = MetaData(bind=self.engine, reflect=True)
                    t = metadata.tables[satel_table_name]
                    mapper(SatelTable, t)
                else:
                    self.logger.error(f'{satel_table_name} not exists')
                    exit(1)

                lats = []
                lons = []
                mingmt = [True for i in range(1, 1441)]
                for satel in self.session.query(SatelTable).yield_per(
                    self.CONFIG['database']['batch_size']['query']):
                    mins = satel.datetime.hour*60 + satel.datetime.minute
                    if mingmt[mins]:
                        lats.append(satel.latitude)
                        lons.append(satel.longitude)
                        mingmt[mins] = False

                map = Basemap(llcrnrlon=-180.0, llcrnrlat=-90.0, urcrnrlon=180.0,
                              urcrnrlat=90.0)
                map.drawcoastlines()
                map.drawmapboundary(fill_color='aqua')
                map.fillcontinents(color='coral', lake_color='aqua')
                map.drawmeridians(np.arange(0, 360, 30))
                map.drawparallels(np.arange(-90, 90, 30))
                map.scatter(lons, lats, latlon=True)
                plt.show()
                breakpoint()

    def match(self):
        self._match_with_cwind()
        self._match_with_sfmr()

    def _match_with_sfmr(self):
        """Add two new columns: hurr_year_name and sfmr_key.
        Then copy corrsponding sfmr data row to satel table.

        """
        # get years in period
        # general sfmr_year table name
        # iterate rows of table
            # match with sfmr location and datetime

        try:
            col_hurrname = Column('roughly_matching_sfmr_hurrname',
                                  String(length=20))
            col_datakey = Column('matching_sfmr_datakey', Integer())
        except Exception as msg:
            breakpoint()
            self.logger.exception(('Error occurred when defining new '
                                   + 'columns about cwind'))
            exit(msg)

        # Roughly figure out the bounday of all hurricans
        hurr_lat = []
        hurr_lon = []

        for hurr in self.session.query(sfmr.HurrSfmr):
            hurr_lat.append(hurr.min_lat)
            hurr_lat.append(hurr.max_lat)
            hurr_lon.append(hurr.min_lon)
            hurr_lon.append(hurr.max_lon)

        min_lat, max_lat = min(hurr_lat), max(hurr_lat)
        min_lon, max_lon = min(hurr_lon), max(hurr_lon)

        # generate satelname_year table name
        for satel_name in self.satel_names:
            for year in self.years:
                satel_table_name = f'{satel_name}_{year}'
                info = f'Matching {satel_table_name} with sfmr'
                self.logger.info(info)

                SatelTable = utils.get_class_by_tablename(self.engine,
                                                          satel_table_name)
                if SatelTable is None:
                    continue

                try:
                    # Check if columns of cwind exists
                    if not hasattr(SatelTable, col_hurrname.name):
                        # Add new columns of cwind
                        utils.add_column(self.engine, satel_table_name,
                                         col_hurrname)
                        utils.add_column(self.engine, satel_table_name,
                                         col_datakey)
                except Exception as msg:
                    self.logger(('Error occurred when checking if '
                                 + 'columns about sfmr exist'))
                    exit(msg)
                # Iterate rows of table
                satel_query = self.session.query(SatelTable).yield_per(
                    self.CONFIG['database']['batch_size']['query'])
                total = satel_query.count()
                match_count = 0
                for idx, row in enumerate(satel_query):

                    print((f'\r{info} ({idx+1}/{total}) '
                           + f'match {match_count}'), end='')
                    # Roughly check whether row is in region of all
                    # hurricanes
                    if (row.latitude < min_lat or row.latitude > max_lat
                        or row.longitude < min_lon or row.longitude > max_lon):
                        continue
                    # match with cwind station location and
                    # corresponding cwind data datetime
                    row.roughly_matching_sfmr_hurrname, \
                            row.matching_hurr_datakey = \
                            self._match_row_with_sfmr(SatelTable, row)

                    if (row.roughly_matching_sfmr_hurrname is not None
                        and row.matching_sfmr_datakey is not None):
                        match_count += 1
                        self.session.commit()

                utils.delete_last_lines()
                print('Done')


    def _match_with_cwind(self):
        """Add two new columns: cwind_station_id and cwind_data_key.
        There copy corresponding cwind data row to satel table.

        """

        try:
            col_stnid = Column('matching_cwind_stnid', String(length=10))
            col_datakey = Column('matching_cwind_datakey', Integer())
        except Exception as msg:
            breakpoint()
            self.logger.exception(('Error occurred when defining new '
                                   + 'columns about cwind'))
            exit(msg)

        stn_lat = []
        stn_lon = []

        for stn in self.session.query(cwind.CwindStation):
            stn_lat.append(stn.latitude)
            stn_lon.append(stn.longitude)

        min_lat, max_lat = min(stn_lat), max(stn_lat)
        min_lon, max_lon = min(stn_lon), max(stn_lon)

        # generate satelname_year table name
        for satel_name in self.satel_names:
            for year in self.years:
                satel_table_name = f'{satel_name}_{year}'
                info = f'Matching {satel_table_name} with cwind'
                self.logger.info(info)

                SatelTable = utils.get_class_by_tablename(self.engine,
                                                          satel_table_name)
                if SatelTable is None:
                    continue

                try:
                    # Check if columns of cwind exists
                    if not hasattr(SatelTable, col_stnid.name):
                        # Add new columns of cwind
                        utils.add_column(self.engine, satel_table_name,
                                         col_stnid)
                        utils.add_column(self.engine, satel_table_name,
                                         col_datakey)
                except Exception as msg:
                    self.logger(('Error occurred when checking if '
                                 + 'columns about cwind exist'))
                    exit(msg)
                # Iterate rows of table
                satel_query = self.session.query(SatelTable).yield_per(
                    self.CONFIG['database']['batch_size']['query'])
                total = satel_query.count()
                match_count = 0
                for idx, row in enumerate(satel_query):

                    print((f'\r{info} ({idx+1}/{total}) '
                           + f'match {match_count}'), end='')
                    if (row.latitude < min_lat or row.latitude > max_lat
                        or row.longitude < min_lon or row.longitude > max_lon):
                        continue
                    # match with cwind station location and
                    # corresponding cwind data datetime
                    row.matching_cwind_stnid, row.matching_cwind_datakey \
                            = self._match_row_with_cwind(SatelTable, row)

                    if (row.matching_cwind_stnid is not None
                        and row.matching_cwind_datakey is not None):
                        match_count += 1
                        self.session.commit()

                utils.delete_last_lines()
                print('Done')


    def _match_row_with_sfmr(self, SatelTable, row):
        """Select the hurricane SFMR data which is matching in space and
        time.

        """
        matching_hurr = None
        matching_data = None

        # Find the roughly matching hurricane candidates in time and space
        matching_hurr_candidate = dict()

        for hurr in self.session.query(sfmr.HurrSfmr):
            if (row.datetime.date < hurr.start_date
                or row.datetime.date > hurr.end_date):
                continue
            if (row.latitude < hurr.min_lat or row.latitude > hurr.max_lat
                or row.longitude < hurr_min_lon
                or row.longitude > hurr_max_lon):
                continue
            matching_hurr_candidate[hurr] = dict()
            matching_hurr_candidate[hurr]['table_name'] = \
                    f'sfmr_{hurr.start_date.year}_{hurr.name}'

            lat_dis = abs(stn.latitude - row.latitude)
            lon_dis = abs(stn.longitude - row.longitude)
            if (lat_dis >= self.spatial_window
                or lon_dis >= self.spatial_window):
                continue
            dis = distance.distance((stn.latitude, stn.longitude),
                                    (row.latitude, row.longitude))
            if dis < min_dis:
                min_dis = dis
                matching_stn = stn

        if not len(matching_hurr_candidate):
            return None, None

        # Find the precisely matching sfmr data record in time and space
        for hurr in matching_hurr_candidate:
            matching_hurr_candidate[hurr]['space_time_dis'] = None
            matching_hurr_candidate[hurr]['datakey'] = None

            SfmrTable = utils.get_class_by_tablename(
                self.engine, matching_hurr_candidate[hurr])
            if SfmrTable is None:
                continue

            min_space_time_dis = 999999999

            for data in self.session.query(SfmrTable).yield_per(
                self.CONFIG['database']['batch_size']['query']):
                # Filter by datetime
                datetime_dis = abs(data.datetime - row.datetime).seconds
                if datetime_dis >= self.temporal_window:
                    continue
                # Filter by latitude and longitude
                lat_dis = abs(stn.latitude - row.latitude)
                lon_dis = abs(stn.longitude - row.longitude)
                if (lat_dis >= self.spatial_window
                    or lon_dis >= self.spatial_window):
                    continue
                # ??? How to calculate space-time distance ???
                location_dis = distance.distance(
                    (data.latitude, data.longitude),
                    (row.latitude, row.longitude))
                space_time_dis = cal_space_time_distance(
                    location_dis, datetime_dis)
                if space_time_dis < min_space_time_dis:
                    min_space_time_dis = space_time_dis
                    matching_hurr_candidate[hurr]['space_time_dis'] =\
                            space_time_dis
                    matching_hurr_candidate[hurr]['datakey'] = data.key

        # Select closest hurricane SFMR data record from candidates
        min_space_time_dis = 999999999
        closest_hurr = None
        for hurr in matching_hurr_candidate:
            if matching_hurr_candidate[hurr]['datakey'] is None:
                continue
            if matching_hurr_candidate[hurr]['space_time_dis']\
               < min_space_time_dis:
                min_space_time_dis = \
                        matching_hurr_candidate[hurr]['space_time_dis']
                closest_hurr = hurr

        if closest_hurr is not None:
            return hurr, matching_hurr_candidate[hurr]['datakey']
        else:
            return None, None

    def _match_row_with_cwind(self, SatelTable, row):
        """Select the cwind station which is the matching in space
        and the cwind data which is the matching in time.

        """
        matching_stn = None
        matching_data = None

        # Find the matching cwind station in space
        min_dis = 99999999

        for stn in self.session.query(cwind.CwindStation):
            lat_dis = abs(stn.latitude - row.latitude)
            lon_dis = abs(stn.longitude - row.longitude)
            if (lat_dis >= self.spatial_window
                or lon_dis >= self.spatial_window):
                continue
            dis = distance.distance((stn.latitude, stn.longitude),
                                    (row.latitude, row.longitude))
            if dis < min_dis:
                min_dis = dis
                matching_stn = stn

        if matching_stn is None:
            return None, None

        # Find the matching cwind data record which belongs to
        # matching cwind station in time
        cwind_data_table_name = f'cwind_{matching_stn.id}'
        CwindData = utils.get_class_by_tablename(
            self.engine, cwind_data_table_name)
        if CwindData is None:
            return None, None

        min_datetime_dis = 99999999

        for data in self.session.query(CwindData).yield_per(
            self.CONFIG['database']['batch_size']['query']):
            datetime_dis = abs(data.datetime - row.datetime).seconds
            if datetime_dis>= self.temporal_window:
                continue
            if datetime_dis < min_datetime_dis:
                min_datetime_dis = datetime_dis
                matching_data = data

        if matching_data is None:
            return matching_stn.id, None
        else:
            return matching_stn.id, matching_data.key

    def _insert_satel(self, read_all):
        if read_all:
            files_path = dict()
            for satel_name in self.satel_names:
                files_path[satel_name] = []
                files_path[satel_name] += [
                    (self.CONFIG[satel_name]['dirs']['bmaps']
                     + x) for x in os.listdir(
                         self.CONFIG[satel_name]['dirs']['bmaps'])
                    if x.endswith('.gz')
                ]
        else:
            files_path = self.downloaded_file_path
            files_path = dict()
            for satel_name in self.satel_names:
                files_path[satel_name] = []
                for file in os.listdir(
                    self.CONFIG[satel_name]['dirs']['bmaps']):
                    pass

        skip_vars = ['mingmt', 'nodata']
        not_null_vars = ['latitude', 'longitude']
        unique_vars = []
        custom_cols = {1: Column('datetime', DateTime(),
                                 nullable=False, unique=False),
                       -1: Column('space_time', String(255),
                                 nullable=False, unique=True)}

        old_files_path = files_path
        files_path = dict()
        for satel_name in old_files_path.keys():
            files_path[satel_name] = []
            for file_path in old_files_path[satel_name]:
                date_= datetime.datetime.strptime(
                    file_path.split('/')[-1].split('_')[1][:8] + '000000',
                    '%Y%m%d%H%M%S').date()
                if utils.check_period(date_, self.period):
                    files_path[satel_name].append(file_path)

        for satel_name in files_path.keys():
            self.logger.info(self.CONFIG[satel_name]\
                             ['prompt']['info']['read'])
            # Create table of particular satellite
            bytemap_file = files_path[satel_name][0]
            if satel_name == 'ascat' or satel_name == 'qscat':
                not_null_vars += ['windspd', 'winddir']
            elif satel_name == 'wsat':
                not_null_vars += ['w-aw', 'wdir'] 

            total = len(files_path[satel_name])
            count = 0

            for file_path in files_path[satel_name]:
                count += 1
                year_str = file_path.split('/')[-1].split('_')[1][:4]
                table_name = '{0}_{1}'.format(satel_name, year_str)

                SatelTable = utils.create_table_from_bytemap(
                    self.engine, satel_name, bytemap_file,
                    table_name, self.session, skip_vars, not_null_vars,
                    unique_vars, custom_cols)

                info = (f'Extracting {satel_name} data '
                        + f'from {file_path.split("/")[-1]}')
                if count > 1:
                    utils.delete_last_lines()
                print(f'\r{info} ({count}/{total})', end='')

                start = time.process_time()
                one_day_records = self._extract_satel_bytemap(satel_name,
                                                              file_path,
                                                              SatelTable)
                end = time.process_time()
                self.logger.debug(f'{info} in {end-start:.2f} s')

                total_sample = one_day_records
                batch_size = \
                        self.CONFIG['database']['batch_size']['insert']
                table_class = SatelTable
                unique_cols = ['space_time']
                session = self.session

                start = time.process_time()
                utils.bulk_insert_avoid_duplicate_unique(
                    total_sample, batch_size, table_class, unique_cols,
                    session)
                end = time.process_time()
                self.logger.debug((f'Bulk inserting {satel_name} data '
                                   + f'into {table_name} '
                                   + f'in {end-start:.2f} s'))

            utils.delete_last_lines()
            print(f'Done')

    def _extract_satel_bytemap(self, satel_name, file_path, SatelTable):
        bm_file = file_path
        table_class = SatelTable
        skip_vars = ['mingmt', 'nodata']
        datetime_func = datetime_from_bytemap
        datetime_col_name = 'datetime'
        missing = self.CONFIG[satel_name]['missing_value']
        valid_func = valid_bytemap
        unique_func = utils.gen_space_time_fingerprint
        unique_col_name = 'space_time'
        lat_name = 'latitude'
        lon_name = 'longitude'
        period = self.period
        region = self.region
        not_null_vars = ['latitude', 'longitude']
        if satel_name == 'ascat' or satel_name == 'qscat':
            not_null_vars += ['windspd', 'winddir']
        elif satel_name == 'wsat':
            not_null_vars += ['w_aw', 'wdir']

        # Not recommend to use utils.extract_bytemap_to_table, because it's
        # very slow due to too much fucntion call
        res = self._extract_bytemap_to_table_2(satel_name, bm_file,
                                               table_class, missing)

        return res

    def _extract_bytemap_to_table_2(self, satel_name, bm_file, table_class,
                                    missing):
        dataset = utils.dataset_of_daily_satel(satel_name, bm_file)
        vars = dataset.variables

        min_lat, max_lat = self.region[0], self.region[1]
        min_lon, max_lon = self.region[2], self.region[3]
        min_lat_idx, max_lat_idx = utils.find_index([min_lat, max_lat], 'lat')
        lat_indices = [x for x in range(min_lat_idx, max_lat_idx+1)]
        min_lon_idx, max_lon_idx = utils.find_index([min_lon, max_lon], 'lon')
        lon_indices = [x for x in range(min_lon_idx, max_lon_idx+1)]

        lat_len = len(lat_indices)
        lon_len = len(lon_indices)
        total = 2 * lat_len * lon_len
        count = 0

        # Store all rows
        whole_table = []

        st = time.time()
        iasc = [0, 1]
        # iasc = 0 (morning, descending passes)
        # iasc = 1 (evening, ascending passes)
        for i in iasc:
            for j in lat_indices:
                for k in lon_indices:
                    count += 1
                    # if count % 2000 == 0:
                    #     progress = float(count)/total*100
                    #     print('\r{:.1f}%'.format(progress), end='')
                    # if not valid_func(vars, i, j, k):
                    if vars['nodata'][i][j][k]:
                        continue
                    table_row = table_class()
                    lat = vars['latitude'][j]
                    lon = vars['longitude'][k]
                    if (not lat or not lon
                        or lat == missing or lon == missing
                        or lat < min_lat or lat > max_lat
                        or lon < min_lon or lon > max_lon):
                        continue
                    # setattr(table_row, lat_name, float(lat))
                    # setattr(table_row, lon_name, float(lon))
                    table_row.latitude = float(lat)
                    table_row.longitude = float(lon)
                    # Set datetime
                    try:
                        mingmt = float(vars['mingmt'][i][j][k])
                        # See note about same mingmt for detail
                        if (mingmt == missing
                            or vars['mingmt'][0][j][k] == \
                            vars['mingmt'][1][j][k]):
                            continue
                        time_str ='{:02d}{:02d}00'.format(
                            *divmod(int(mingmt), 60))
                        bm_file_name = bm_file.split('/')[-1]
                        date_str = bm_file_name.split('_')[1][:8]

                        if time_str.startswith('24'):
                            date_ = datetime.datetime.strptime(
                                date_str + '000000', '%Y%m%d%H%M%S').date()
                            time_ = datetime.time(0, 0, 0)
                            date_ = date_ + datetime.timedelta(days=1)
                            datetime_ = datetime.datetime.combine(
                                date_, time_)
                        else:
                            datetime_ = datetime.datetime.strptime(
                                date_str + time_str, '%Y%m%d%H%M%S')
                    except Exception as msg:
                        breakpoint()
                        exit(msg)
                    # Period check
                    if not datetime_ or not utils.check_period(datetime_,
                                                         self.period):
                        continue
                    table_row.datetime = datetime_

                    table_row.space_time = '%s %f %f' % (datetime_, lat, lon)

                    valid = True
                    table_row.land = bool(vars['land'][i][j][k])
                    table_row.ice = bool(vars['ice'][i][j][k])
                    if satel_name == 'ascat' or satel_name == 'qscat':
                        table_row.windspd = float(vars['windspd'][i][j][k])
                        table_row.winddir = float(vars['winddir'][i][j][k])
                        if (table_row.windspd is None 
                            or table_row.winddir is None
                            or table_row.windspd == missing
                            or table_row.winddir == missing):
                            continue
                        table_row.scatflag = float(vars['scatflag'][i][j][k])
                        table_row.radrain = float(vars['radrain'][i][j][k])
                        if satel_name == 'ascat':
                            table_row.sos = float(vars['sos'][i][j][k])
                    elif satel_name == 'wsat':
                        table_row.w_aw = float(vars['w-aw'][i][j][k])
                        table_row.wdir = float(vars['wdir'][i][j][k])
                        if (table_row.w_aw is None 
                            or table_row.wdir is None
                            or table_row.w_aw == missing
                            or table_row.wdir == missing):
                            continue
                        table_row.vapor = float(vars['vapor'][i][j][k])
                        table_row.cloud = float(vars['cloud'][i][j][k])
                        table_row.rain = float(vars['rain'][i][j][k])
                        table_row.w_lf = float(vars['w-lf'][i][j][k])
                        table_row.w_mf = float(vars['w-mf'][i][j][k])
                    else:
                        sys.exit('satel_name is wrong.')

                    if valid:
                        whole_table.append(table_row)

        return whole_table

    def _read_satel_dataset(self, satel_name, dataset, missing_val=-999.0):
        min_lat_index, max_lat_index = self._find_index(
            [self.region[0], self.region[1]], 'lat')
        lat_indices = [x for x in range(min_lat_index, max_lat_index+1)]
        min_lon_index, max_lon_index = self._find_index(
            [self.region[2], self.region[3]], 'lon')
        lon_indices = [x for x in range(min_lon_index, max_lon_index+1)]

    def _download_single_satel(self, config, satel_name, period):
        """Download ASCAT/QucikSCAT/Windsat data in specified date range.

        """
        info = config['prompt']['info']['download']
        self.logger.info(info)
        start_date = period[0].date()
        end_date = period[1].date()
        data_url = config['urls']
        file_suffix = config['data_suffix']
        save_dir = config['dirs']['bmaps']
        missing_dates_file = config['files_path']['missing_dates']

        utils.set_format_custom_text(config['data_name_length'])
        if os.path.exists(missing_dates_file):
            with open(missing_dates_file, 'rb') as fr:
                missing_dates = pickle.load(fr)
        else:
            missing_dates = set()

        os.makedirs(save_dir, exist_ok=True)
        delta_date = end_date - start_date

        total = delta_date.days + 1
        count = 0

        for i in range(delta_date.days + 1):
            count += 1
            print(f'\r({count}/{total})', end='')
            self.logger.debug(info)

            date_ = start_date + datetime.timedelta(days=i)
            if date_ in missing_dates:
                continue
            file_name = '%s_%04d%02d%02d%s' % (
                satel_name, date_.year, date_.month, date_.day, file_suffix)
            file_url = '%sy%04d/m%02d/%s' % (
                data_url, date_.year, date_.month, file_name)
            if not utils.url_exists(file_url):
                print('Missing date: ' + str(date_))
                print(file_url)
                missing_dates.add(date_)
                continue

            file_path = save_dir + file_name

            if not utils.ready_for_download(file_url, file_path):
                return
            utils.download(file_url, file_path)
            self.downloaded_file_path[satel_name].append(file_path)

        utils.delete_last_lines()
        print('Done')

        with open(missing_dates_file, 'wb') as fw:
            pickle.dump(missing_dates, fw)

def datetime_from_bytemap(bm_file_path, vars, i, j, k, missing):
    bm_file_name = bm_file_path.split('/')[-1]
    date_str = bm_file_name.split('_')[1][:8]
    mingmt = int(vars['mingmt'][i][j][k])
    time_str = '{:02d}{:02d}00'.format(*divmod(mingmt, 60))

    datetime_ = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

    return datetime_

def valid_bytemap(vars, i, j, k):
    if vars['nodata'][i][j][k]:
        return False
    return True

def row2dict(row):
    d = row.__dict__
    d.pop('_sa_instance_state', None)

    return d
