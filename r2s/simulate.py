import calendar
import copy
import datetime
import logging
import os
import pickle
import statistics
import string
import time

import matplotlib.pyplot as plt
from global_land_mask import globe
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import pygrib
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import or_
import netCDF4

import ccmp
import era5
import satel_scs
import utils
import match_era5_smap
import compare_tc

Base = declarative_base()

class TCSimulator(object):

    def __init__(self, CONFIG, period, region, basin, passwd,
                 save_disk, simulate_instructions):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.save_disk = save_disk
        self.engine = None
        self.session = None
        self.simulate_instructions = list(set(simulate_instructions))
        self.basin = basin

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.spa_resolu = dict()

        self.spa_resolu['era5'] = dict()
        self.spa_resolu['era5']['atm'] = self.CONFIG['era5'][
            'spatial_resolution']
        self.spa_resolu['era5']['ocean'] = self.CONFIG['era5'][
            'ocean_spatial_resolution']

        self.lats = dict()
        self.lons = dict()

        self.lats['era5'] = [y * self.spa_resolu['era5']['atm'] - 90
                             for y in range(721)]
        self.lons['era5'] = [x * self.spa_resolu['era5']['atm']
                             for x in range(1440)]

        self.spa_resolu['smap'] = self.CONFIG['rss'][
            'spatial_resolution']
        self.lats['smap'] = [
            y * self.spa_resolu['smap'] - 89.875 for y in range(720)]
        self.lons['smap'] = [
            x * self.spa_resolu['smap'] + 0.125 for x in range(1440)]

        self.reg_edge = self.CONFIG['regression']['edge_in_degree']

        self.half_reg_edge = self.reg_edge / 2
        self.half_reg_edge_grid_intervals = int(
            self.half_reg_edge / self.spa_resolu['era5']['atm'])

        self.pres_lvls = self.CONFIG['era5']['pres_lvls']

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.zorders = self.CONFIG['plot']['zorders']['compare']

        utils.reset_signal_handler()
        self.tc_names = []
        for name in self.simulate_instructions:
            self.tc_names.append(name.upper())

        self.simulate_smap_windspd()

    """
+---------------+-----------+---------------------+-----------------------------------+
| sid           | name      | start_datetime      | intensity_change_percent_per_hour |
+---------------+-----------+---------------------+-----------------------------------+
| 2015193N35285 | CLAUDETTE | 2015-07-13 06:00:00 |                         0.0555556 |
| 2017170N08310 | BRET      | 2017-06-20 00:00:00 |                            0.0625 |
| 2017219N16279 | FRANKLIN  | 2017-08-09 00:00:00 |                            0.0625 |
| 2017249N22263 | KATIA     | 2017-09-06 12:00:00 |                         0.0740741 |
| 2017258N10337 | LEE       | 2017-09-24 00:00:00 |                              0.05 |
| 2017260N12310 | MARIA     | 2017-09-16 12:00:00 |                         0.0555556 |
| 2017260N12310 | MARIA     | 2017-09-18 18:00:00 |                         0.0530303 |
| 2018246N22283 | GORDON    | 2018-09-03 06:00:00 |                         0.0555556 |
| 2018246N22283 | GORDON    | 2018-09-03 09:00:00 |                          0.126984 |
| 2018265N35315 | LESLIE    | 2018-09-25 12:00:00 |                         0.0555556 |
+---------------+-----------+---------------------+-----------------------------------+
"""

    def simulate_smap_windspd(self):
        self.logger.info((
            f"""Comparing wind speed from different sources"""))
        # Get IBTrACS table
        table_name = self.CONFIG['ibtracs']['table_name'][
            self.basin]
        IBTrACS = utils.get_class_by_tablename(self.engine,
                                               table_name)
        query_obj = self.session.query(IBTrACS).filter(
            IBTrACS.date_time >= self.period[0],
            IBTrACS.date_time <= self.period[1]
        )
        in_expression = IBTrACS.name.in_(self.tc_names)
        tc_query = query_obj.filter(in_expression)
        total = tc_query.count()

        # Expand period
        if total < 2:
            self.logger.warning('Expand period')
            query_obj = self.session.query(IBTrACS).filter(
                IBTrACS.date_time >= (self.period[0]
                                      - datetime.timedelta(
                                          seconds=3600*3)),
                IBTrACS.date_time <= (self.period[1]
                                      + datetime.timedelta(
                                          seconds=3600*3))
            )
            in_expression = IBTrACS.name.in_(self.tc_names)
            tc_query = query_obj.filter(in_expression)
            total = tc_query.count()
            if total < 2:
                self.logger.error('Too few TCs')
                exit(1)

        # Filter TCs during period
        for idx, tc in enumerate(tc_query):
            if tc.name not in self.tc_names:
                continue
            converted_lon = utils.longitude_converter(tc.lon,
                                                      '360', '-180')
            if bool(globe.is_land(tc.lat, converted_lon)):
                continue
            success = False

            if idx < total - 1:
                next_tc = tc_query[idx + 1]
                if tc.sid == next_tc.sid:
                    if (tc.date_time >= self.period[1]
                            or next_tc.date_time <= self.period[0]):
                        continue
                    print(f'Simulating {tc.date_time} - {next_tc.date_time}')

                    self.simulate_between_two_tcs(tc, next_tc)

        print('Done')

    def simulate_between_two_tcs(self, tc, next_tc):
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are far away in time
        if delta.days:
            return False
        hours = int(delta.seconds / 3600)
        # Time interval between two TCs are less than one hour
        if not hours:
            return False

        # `tight_hours` is for the case that period totally falls in
        # the interval of two TC records and is shorted than the
        # interval.
        if (self.period[0] > tc.date_time
                or self.period[1] < next_tc.date_time):
            tight_hours = int((self.period[1] - self.period[0]).seconds / 3600)
        else:
            tight_hours = hours

        skip_hour_indices = []
        for i, h in enumerate(range(hours)):
            interped_tc = utils.interp_tc(self, h, tc, next_tc)
            if (interped_tc.date_time < self.period[0]
                    or interped_tc.date_time > self.period[1]):
                skip_hour_indices.append(i)

        subplots_row, subplots_col, fig_size = \
            utils.get_subplots_row_col_and_fig_size(tight_hours)
        fig, axes = plt.subplots(subplots_row, subplots_col,
                                 figsize=fig_size)

        hours_max_windspd = -1
        ax_idx = 0
        for i, h in enumerate(range(hours)):
            if i in skip_hour_indices:
                continue
            try:
                interped_tc = utils.interp_tc(self, h, tc, next_tc)
                if isinstance(axes, np.ndarray):
                    ax = axes.flat[ax_idx]
                else:
                    ax = axes
            except Exception as msg:
                breakpoint()
                exit(msg)

            success, hourly_max_windspd = self.simulate_hourly(
                interped_tc, fig, ax)
            if success:
                hours_max_windspd = max(hours_max_windspd,
                                        hourly_max_windspd)
                ax_idx += 1
        if hourly_max_windspd == -1:
            return

        ax_idx = 0
        for i, h in enumerate(range(hours)):
            if i in skip_hour_indices:
                continue
            try:
                interped_tc = utils.interp_tc(self, h, tc, next_tc)
                if isinstance(axes, np.ndarray):
                    ax = axes.flat[ax_idx]
                else:
                    ax = axes
            except Exception as msg:
                breakpoint()
                exit(msg)

            success, hourly_max_windspd = self.simulate_hourly(
                interped_tc, fig, ax, hours_max_windspd, ax_idx)

            if success:
                ax_idx += 1
                print((f"""Simulating SMAP windspd """
                       f"""of TC {interped_tc.name} on """
                       f"""{interped_tc.date_time}"""))
            else:
                print((f"""Skiping simulating SMAP windspd """
                       f"""of TC {interped_tc.name} """
                       f"""on {interped_tc.date_time}"""))

        fig.tight_layout(pad=0.1)

        dt_str = (f"""{tc.date_time.strftime('%Y_%m%d_%H%M')}"""
                  f"""_"""
                  f"""{next_tc.date_time.strftime('_%H%M')}""")
        fig_dir = self.CONFIG['result']['dirs']['fig']['simulation']

        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'{dt_str}_{tc.name}.png'
        plt.savefig(f'{fig_dir}{fig_name}', dpi=600)
        plt.clf()

    def simulate_hourly(self, tc, fig, ax, hours_max_windspd=None,
                        hour_idx=None):
        success, smap_lons, smap_lats = utils.get_smap_lonlat(self, tc)
        if not success:
            return False, None
        del success
        # North, South, West, East
        draw_region = [min(smap_lats), max(smap_lats),
                       min(smap_lons), max(smap_lons)]

        CompareTC = compare_tc.TCComparer(self.CONFIG, self.period,
                                          self.region, self.basin,
                                          self.db_root_passwd, False,
                                          ['sfmr', 'smap_prediction'],
                                          draw_sfmr=False,
                                          work=False)
        try:
            # Get lons, lats, windspd
            success, lons, lats, windspd, mesh, diff_mins = \
                CompareTC.get_sources_xyz_matrix(
                    'smap_prediction', tc, smap_lons,
                    smap_lats)
            if not success:
                return False, None
            if hours_max_windspd is None:
                return True, windspd.max()
        except Exception as msg:
            breakpoint()
            exit(msg)

        accurate_dt = tc.date_time
        subplot_title_suffix = (
            f"""{accurate_dt.strftime('%H%M UTC %d %b %Y')} """
        )
        # Draw windspd
        print(tc.date_time)
        try:
            fontsize = 20
            utils.draw_windspd(self, fig, ax, tc.date_time,
                               lons, lats, windspd,
                               hours_max_windspd,
                               mesh, draw_contour=True,
                               custom=True,
                               region=draw_region)
            ax.text(0.1, 0.95,
                    f'{string.ascii_lowercase[hour_idx]})',
                    transform=ax.transAxes, fontsize=20,
                    fontweight='bold', va='top', ha='right')
            ax.set_title((
                          f"""Simulated SMAP Wind """
                          f"""{subplot_title_suffix} """
                          f"""{tc.name}"""),
                         size=15)
        except Exception as msg:
            breakpoint()
            exit(msg)

        return True, 'Done'
