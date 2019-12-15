import datetime
import logging
import os
import pickle
import string

import matplotlib.pyplot as plt
from global_land_mask import globe

import ccmp
import era5
import satel_scs
import utils

class TCComparer(object):

    def __init__(self, CONFIG, period, region, passwd, save_disk):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.save_disk = save_disk
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)

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

        self.sources = ['era5', 'smap']
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
            success = self.compare_with_one_tc_record(tc)
            if success:
                print((f"""Comparing {sources_str} with IBTrACS record """
                       f"""of TC {tc.name} on {tc.date_time}"""))
            else:
                print((f"""Skiping comparsion of {sources_str} with """
                       f"""IBTrACS record of TC {tc.name} """
                       f"""on {tc.date_time}"""))
        print('Done')

    def compare_with_one_tc_record(self, tc):
        subplots_row, subplots_col = utils.get_subplots_row_col(
            len(self.sources))
        fig, axs = plt.subplots(subplots_row, subplots_col,
                                figsize=(15, 7), sharey=True)
        axs_list = []
        for idx, src in enumerate(self.sources):
            if subplots_row == 1:
                col_idx = idx % subplots_col
                ax = axs[col_idx]
            elif subplots_row > 1:
                row_idx = int(idx / subplots_col)
                col_idx = idx % subplots_col
                ax = axs[row_idx][col_idx]

            axs_list.append(ax)

        lons = dict()
        lats = dict()
        windspd = dict()
        radii_area = dict()
        mesh = dict()

        max_windspd = -1
        for idx, src in enumerate(self.sources):
            ax = axs_list[idx]

            # Get lons, lats, windspd
            success, lons[src], lats[src], windspd[src], mesh[src] = \
                    self.get_sources_xyz(src, tc, ax)
            if not success:
                return False
            # Get max windspd
            if windspd[src].max() > max_windspd:
                max_windspd = windspd[src].max()
            # Draw wind radii
            radii_area[src] = utils.draw_ibtracs_radii(ax, tc,
                                                       self.zorders)

        # Draw windspd
        for idx, src in enumerate(self.sources):
            ax = axs_list[idx]

            utils.draw_windspd(self, fig, ax, tc.date_time,
                               lons[src], lats[src], windspd[src],
                               max_windspd, mesh[src])
            ax.text(-0.1, 1.025, f'({string.ascii_lowercase[idx]})',
                    transform=ax.transAxes, fontsize=16,
                    fontweight='bold', va='top', ha='right')

        fig.subplots_adjust(wspace=0.4)

        dt_str = tc.date_time.strftime('%Y_%m%d_%H%M')
        fig_dir = (f"""{self.CONFIG['result']['dirs']['fig']['root']}"""
                   f"""ibtracs_vs""")
        for idx, src in enumerate(self.sources):
            if idx < len(self.sources) - 1:
                fig_dir = f"""{fig_dir}_{src}_and"""
            else:
                fig_dir = f"""{fig_dir}_{src}/"""

        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'{dt_str}_{tc.name}.png'
        plt.savefig(f'{fig_dir}{fig_name}')
        plt.clf()

        return True

    def get_sources_xyz(self, src, tc, ax):
        if src == 'ccmp':
            return self.get_ccmp_xyz(tc, ax)
        elif src == 'era5':
            return self.get_era5_xyz(tc, ax)
        elif src == 'smap':
            return self.get_smap_xyz(tc, ax)

    def get_smap_xyz(self, tc, ax):
        satel_manager = satel_scs.SCSSatelManager(
            self.CONFIG, self.period, self.region, self.db_root_passwd,
            save_disk=self.save_disk, work=False)
        smap_file_path = satel_manager.download('smap', tc.date_time)

        lons, lats, windspd = utils.get_xyz_of_smap_windspd(
            smap_file_path, tc.date_time, self.region)

        if (not utils.satel_data_cover_tc_center(lons, lats, windspd, tc)
            or windspd is None):
            return False, None, None, None, None
        else:
            return True, lons, lats, windspd, utils.if_mesh(lons)

    def get_ccmp_xyz(self, tc, ax):
        ccmp_manager = ccmp.CCMPManager(self.CONFIG, self.period,
                                        self.region, self.db_root_passwd,
                                        # work_mode='fetch')
                                        work_mode='')
        ccmp_file_path = ccmp_manager.download_ccmp_on_one_day(
            tc.date_time)

        lons, lats, windspd = utils.get_xyz_of_ccmp_windspd(
            ccmp_file_path, tc.date_time, self.region)

        if windspd is not None:
            return True, lons, lats, windspd, utils.if_mesh(lons)
        else:
            self.logger.info((f"""No CCMP data on {tc.date_time}"""))
            return False, None, None, None, None

    def get_era5_xyz(self, tc, ax):
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = era5_manager.download_surface_vars_of_whole_day(
                 'single_levels', 'surface_wind', tc.date_time)
        lons, lats, windspd = utils.get_xyz_of_era5_windspd(
            era5_file_path, 'single_levels', tc.date_time, self.region)

        return True, lons, lats, windspd, utils.if_mesh(lons)
