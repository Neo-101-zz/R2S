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

import ccmp
import era5
import utils

Base = declarative_base()

class TCComparer(object):

    def __init__(self, CONFIG, period, region, passwd):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.sources = ['ccmp', 'era5', 'ascat', 'wsat', 'amsr2',
                        'smap', 'sentinel_1']

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

        utils.reset_signal_handler()

        self._get_region_corners_indices()

        self.compare_sources()

    def _get_region_corners_indices(self):
        self.lat1_index = self.grid_pts['ccmp']['lat'].index(
            self.lat1 + 0.5 * self.spa_resolu['ccmp'])
        self.lat2_index = self.grid_pts['ccmp']['lat'].index(
            self.lat2 - 0.5 * self.spa_resolu['ccmp'])
        self.lon1_index = self.grid_pts['ccmp']['lon'].index(
            self.lon1 + 0.5 * self.spa_resolu['ccmp'])
        self.lon2_index = self.grid_pts['ccmp']['lon'].index(
            self.lon2 - 0.5 * self.spa_resolu['ccmp'])

    def compare_sources(self):
        self.logger.info((f"""Comparing CCMP and ERA5 with IBTrACS"""))
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
            # Draw windspd from CCMP, ERA5, Interium
            # and several satellites
            print((f"""Comparing CCMP and ERA5 with IBTrACS record of """
                   f"""TC {tc.name} on {tc.date_time}"""), end='')
            self.compare_with_one_tc_record(tc)
            utils.delete_last_lines()
        print('Done')

    def compare_with_one_tc_record(self, tc):
        dt = tc.date_time
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
        lons = dict()
        lats = dict()
        windspd = dict()

        ax_ccmp = axs[0]
        ax_era5 = axs[1]

        ccmp_manager = ccmp.CCMPManager(self.CONFIG, self.period,
                                        self.region, self.db_root_passwd,
                                        # work_mode='fetch')
                                        work_mode='')
        ccmp_file_path = ccmp_manager.download_ccmp_on_one_day(dt)
        lons['ccmp'], lats['ccmp'], windspd['ccmp'] = \
                utils.get_xyz_of_ccmp_windspd(ccmp_file_path, dt,
                                              self.region)

        if windspd['ccmp'] is None:
            self.logger.info((f"""No CCMP data on {dt}"""))
            return

        # CCMP = ccmp_manager.create_scs_ccmp_table(dt.date())
        # lons['ccmp'], lats['ccmp'], windspd['ccmp'] = \
        #         ccmp_manager.get_xyz_of_ccmp_windspd(CCMP, dt)


        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False, save_disk=False,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = era5_manager.download_surface_vars_of_whole_day(
                 'surface_wind', dt)
        lons['era5'], lats['era5'], windspd['era5'] = \
                utils.get_xyz_of_era5_windspd(era5_file_path, dt,
                                              self.region)

        radii_area = utils.draw_ibtracs_radii(ax_ccmp, tc, self.zorders)
        radii_area = utils.draw_ibtracs_radii(ax_era5, tc, self.zorders)

        max_windspd = max(windspd['ccmp'].max(), windspd['era5'].max())

        utils.draw_windspd(self, fig, ax_ccmp, dt, lons['ccmp'],
                           lats['ccmp'], windspd['ccmp'], max_windspd,
                           mesh=True)
        utils.draw_windspd(self, fig, ax_era5, dt, lons['era5'],
                           lats['era5'], windspd['era5'], max_windspd,
                           mesh=False)

        dt_str = dt.strftime('%Y_%m%d_%H%M')
        fig_name = f'ibtracs_vs_ccmp_vs_era5_{dt_str}_{tc.name}.png'
        fig_dir = self.CONFIG['result']['dirs']['fig']
        plt.show()
        # plt.savefig(f'{fig_dir}{fig_name}')
        plt.clf()

