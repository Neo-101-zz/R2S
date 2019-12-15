import datetime
import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import grid
import utils

Base = declarative_base()

class CoverageManager(object):

    def __init__(self, CONFIG, period, region, passwd):
        self.logger = logging.getLogger(__name__)
        self.satel_names = ['ascat', 'wsat', 'amsr2', 'smap',
                            'sentinel_1']

        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.edge = self.CONFIG['rss']['subset_edge_in_degree']
        self.spa_resolu = self.CONFIG['rss']['spatial_resolution']

        self.grid_pts = dict()

        self.grid_pts['rss'] = dict()
        self.grid_pts['rss']['lat'] = [
            y * self.spa_resolu - 89.875 for y in range(
                self.CONFIG['rss']['lat_grid_points_number'])
        ]
        self.grid_pts['rss']['lon'] = [
            x * self.spa_resolu + 0.125 for x in range(
                self.CONFIG['rss']['lon_grid_points_number'])
        ]

        self.grid_pts['era5'] = dict()
        self.grid_pts['era5']['lat'] = [
            y * self.spa_resolu - 90 for y in range(
                self.CONFIG['era5']['lat_grid_points_number'])
        ]
        self.grid_pts['era5']['lon'] = [
            x * self.spa_resolu for x in range(
                self.CONFIG['era5']['lon_grid_points_number'])
        ]

        self.missing_value = dict()
        for satel_name in self.satel_names:
            if satel_name != 'smap':
                self.missing_value[satel_name] = \
                        self.CONFIG[satel_name]['missing_value']
            else:
                self.missing_value[satel_name] = dict()
                self.missing_value[satel_name]['minute'] = \
                        self.CONFIG[satel_name]['missing_value']['minute']
                self.missing_value[satel_name]['wind'] = \
                        self.CONFIG[satel_name]['missing_value']['wind']

        utils.setup_database(self, Base)

        self.grid_df = pd.read_sql('SELECT * FROM grid', self.engine)
        self.grid_lons = self.grid_df['lon']
        self.grid_lats = self.grid_df['lat']
        self.grid_x = self.grid_df['x']
        self.grid_y = self.grid_df['y']

        self.zorders = self.CONFIG['plot']['zorders']['scs_basemap']
        self.draw_coverage()

    def draw_coverage(self):
        start, end = self.period[0], self.period[1]
        # Generate all entire hours in subperiod
        # `hourly_dt` is a list which has all chronological hours
        # that cover the whole subperoid
        # e.g. subperiod is from 2019-10-30 12:34:11 to
        # 2019-11-01 10:29:56, the hourly_dt should be
        # a datetime list starting with 2019-10-30 12:00:00 and
        # ending with 2019-11-01 11:00:00
        hourly_dt = self.gen_hourly_dt_in_subperiod(start, end)
        # Generate all hours during subperiod

        # Subperiod is shorter than one day

        # Subperiod is longer than one day

        for i in range(len(hourly_dt) - 1):
            this_hour = hourly_dt[i]
            next_hour = hourly_dt[i + 1]
            coverage = dict()
            all_satels_null = True

            for satel_name in self.satel_names:
                tablename = utils.gen_satel_era5_tablename(satel_name,
                                                           this_hour)
                SatelERA5 = utils.get_class_by_tablename(self.engine,
                                                         tablename)
                coverage[satel_name] = self.get_satel_coverage(
                    satel_name, SatelERA5, this_hour, next_hour)
                valid_pts_num, lons, lats, windspd = coverage[satel_name]

                if len(lons) >= 2 and len(lats) >= 2:
                    # 
                    all_satels_null = False

            if all_satels_null:
                self.logger.info((f"""All satellites have no data """
                                  f"""from {this_hour} to """
                                  f"""{next_hour}"""))
                continue

            self.draw_coverage_of_all_satels(this_hour, next_hour,
                                             coverage)

    def get_satel_coverage(self, satel_name, SatelERA5, this_hour,
                           next_hour):
        self.logger.info((f"""Getting coverge of {satel_name} """
                          f"""from {this_hour} to {next_hour}"""))

        satel_coverage = dict()
        satel_coverage['lon'] = []
        satel_coverage['lat'] = []
        satel_coverage['windspd'] = []

        satel_windspd_col_name = {
            'ascat': 'windspd',
            'wsat': 'w_aw',
            'amsr2': 'wind_lf',
            'smap': 'windspd',
            'sentinel_1': 'windspd'
        }

        grid_lons_lats = dict()

        for name in ['lons', 'lats']:
            pickle_path = self.CONFIG['grid']['pickle'][name]
            with open(pickle_path, 'rb') as f:
                grid_lons_lats[name] = pickle.load(f)

        query_for_count = self.session.query(SatelERA5).filter(
            SatelERA5.satel_datetime >= this_hour,
            SatelERA5.satel_datetime < next_hour)
        total = query_for_count.count()
        del query_for_count
        count = 0

        if not total:
            return 0, [], [], 0

        min_lon, max_lon = 999, -999
        min_lat, max_lat = 999, -999

        for row in self.session.query(SatelERA5).filter(
            SatelERA5.satel_datetime >= this_hour,
            SatelERA5.satel_datetime < next_hour).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            count += 1
            print(f'\rTraversing data: {count}/{total}', end='')

            lon = grid_lons_lats['lons'][row.x]
            satel_coverage['lon'].append(lon)
            if lon < min_lon:
                min_lon = lon
            if lon > max_lon:
                max_lon = lon

            lat = grid_lons_lats['lats'][row.y]
            satel_coverage['lat'].append(lat)
            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat

            satel_coverage['windspd'].append(getattr(
                row, satel_windspd_col_name[satel_name]))

        print('Done')
        utils.delete_last_lines()

        if min_lon > max_lon or min_lat > max_lat:
            return 0, [], [], 0

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

        if satel_name != 'sentinel_1':
            for i in range(total):
                count += 1
                try:
                    lon_idx = lons.index(satel_coverage['lon'][i])
                    lat_idx = lats.index(satel_coverage['lat'][i])
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
                                satel_coverage['windspd'][i]
        else:
            for i in range(total):
                count += 1
                lon_idx = lons.index(satel_coverage['lon'][i])
                lat_idx = lats.index(satel_coverage['lat'][i])

                windspd[lat_idx][lon_idx] = satel_coverage['windspd'][i]

        return total, lons, lats, windspd

    def draw_coverage_of_all_satels(self, this_hour, next_hour,
                                    coverage):
        self.logger.info((f"""Drawing coverage of all satellites """
                          f"""from {this_hour} to {next_hour}"""))

        subplots = {
            'ascat': 231, 'wsat': 232,
            'amsr2': 233, 'smap': 234,
            'sentinel_1': 235
        }

        fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True)

        max_windspd = 0
        for satel_name in self.satel_names:
            windspd = coverage[satel_name][-1]
            if not np.count_nonzero(windspd):
                continue

            if windspd.max() > max_windspd:
                max_windspd = windspd.max()

        for idx, satel_name in enumerate(self.satel_names):
            ax = axs[int(idx / 3)][idx % 3]
            # ax = fig.add_subplot(subplots[satel_name], aspect='equal')
            ax.axis([self.lon1, self.lon2, self.lat1, self.lat2])

            utils.draw_SCS_basemap(self, ax)

            valid_pts_num, lons, lats, windspd = coverage[satel_name]
            ax.set_title(f'{satel_name}: {valid_pts_num}')

            if len(lons) < 2 and len(lats) < 2:
                continue

            self._draw_satel_windspd(fig, ax, lons, lats, windspd,
                                     max_windspd)

        start_dt = this_hour.strftime('%Y_%m%d_%H%M')
        end_dt = next_hour.strftime('%Y_%m%d_%H%M')
        fig_name = f'{start_dt}_{end_dt}.png'
        fig_dir = self.CONFIG['result']['dirs']['fig']\
                ['satellite_coverage']
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f'{fig_dir}{fig_name}')
        plt.clf()

    def _draw_coverage_basemap(self, ax):

        map = Basemap(llcrnrlon=self.lon1, llcrnrlat=self.lat1,
                      urcrnrlon=self.lon2, urcrnrlat=self.lat2,
                      ax=ax)

        map.drawcoastlines(zorder=self.zorders['coastlines'])
        map.drawmapboundary(fill_color='white',
                            zorder=self.zorders['mapboundary'])
        map.fillcontinents(color='grey', lake_color='white',
                           zorder=self.zorders['continents'])

        map.drawmeridians(np.arange(100, 125, 5),
                          labels=[1, 0, 0, 1])
        map.drawparallels(np.arange(0, 30, 5),
                          labels=[1, 0 , 0, 1])

    def _draw_satel_windspd(self, fig, ax, lons, lats, windspd,
                            max_windspd):
        X, Y = np.meshgrid(lons, lats)
        Z = windspd

        # windspd_levels = [5*x for x in range(1, 15)]
        windspd_levels = np.linspace(0.01, max_windspd, 10)

        # cs = ax.contour(X, Y, Z, levels=windspd_levels,
        #                 zorder=self.zorders['contour'], colors='k')
        # ax.clabel(cs, inline=1, colors='k', fontsize=10)
        cf = ax.contourf(X, Y, Z, levels=windspd_levels,
                         zorder=self.zorders['contourf'],
                         cmap=plt.cm.rainbow)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(cf, cax=cax, orientation='vertical', format='%.1f')

    def gen_hourly_dt_in_subperiod(self, start, end):
        hourly_dt = []
        start = utils.backtime_to_last_entire_hour(start)
        end = utils.forwardtime_to_next_entire_hour(end)

        delta_dt = end - start
        # Generate all hours during subperiod
        the_hour = start
        while(the_hour <= end):
            hourly_dt.append(the_hour)
            the_hour = the_hour + datetime.timedelta(hours=1)

        return hourly_dt
