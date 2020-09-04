"""Download and read ERA5 reanalysis data from ECMWF.

"""
import datetime
import logging
import math
import os
import sys
import time

import cdsapi
import pygrib
import numpy as np
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Column
from sqlalchemy.schema import Table, MetaData
from sqlalchemy.orm import mapper
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
from global_land_mask import globe

import ibtracs
import utils

Base = declarative_base()
DEGREE_OF_ONE_NMILE = float(1)/60
KM_OF_ONE_NMILE = 1.852
KM_OF_ONE_DEGREE = KM_OF_ONE_NMILE / DEGREE_OF_ONE_NMILE

class ERA5Manager(object):
    """Manage features of ERA5 that are not related to other data
    sources except TC table from IBTrACS.

    """
    def __init__(self, CONFIG, period, region, passwd, work,
                 save_disk, work_mode, vars_mode):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.save_disk = save_disk

        self.logger = logging.getLogger(__name__)

        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]
        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.main_hours = self.CONFIG['era5']['main_hours']
        self.edge = self.CONFIG['regression']['edge_in_degree']

        self.cdsapi_client = cdsapi.Client()
        self.vars = self.CONFIG['era5']['vars']
        self.surface_pres_lvl = '1000'

        self.spa_resolu = self.CONFIG['era5']['spatial_resolution']
        self.lat_grid_points = [y * self.spa_resolu - 90 for y in range(
            self.CONFIG['era5']['lat_grid_points_number'])]
        self.lon_grid_points = [x * self.spa_resolu for x in range(
            self.CONFIG['era5']['lon_grid_points_number'])]
        # Size of threeD grid points around TC center
        self.threeD_grid = dict()
        self.threeD_grid['lat_axis'] = self.threeD_grid['lon_axis'] = int(
            self.edge/self.spa_resolu) + 1
        self.threeD_grid['height'] = len(
            self.CONFIG['era5']['pres_lvls'])

        self.twoD_grid = dict()
        self.twoD_grid['lat_axis'] = self.twoD_grid['lon_axis'] = int(
            self.edge/self.spa_resolu) + 1

        self.wind_radii = self.CONFIG['wind_radii']

        self.grid_pts = dict()

        self.grid_pts['era5'] = dict()
        self.grid_pts['era5']['lat'] = [
            y * self.spa_resolu - 90 for y in range(
                self.CONFIG['era5']['lat_grid_points_number'])
        ]
        self.grid_pts['era5']['lon'] = [
            x * self.spa_resolu for x in range(
                self.CONFIG['era5']['lon_grid_points_number'])
        ]

        self.zorders = self.CONFIG['plot']['zorders']['compare']

        utils.setup_database(self, Base)

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        if work:
            # work_mode: 'scs' / 'tc'
            # vars_mode: 'surface_all_vars' / 'threeD' / 'surface_wind'
            self.download_and_read(work_mode, vars_mode)
            # self.download_and_read('tc', 'surface_all_vars')

    # def get_era5_columns(self, satel_era5_time_diff=True):
    def get_era5_columns(self, satel_era5_time_diff=True, tgt_name=None):
        cols = []

        if satel_era5_time_diff:
            if tgt_name is None:
                cols.append(Column('satel_era5_diff_mins', Integer,
                                   nullable=False))
            else:
                cols.append(Column(f'{tgt_name}_era5_diff_mins', Integer,
                                   nullable=False))

        for var in self.vars['grib']['reanalysis_single_levels']\
                   ['tc']:
            cols.append(Column(var, Float, nullable=False))

        for var in self.vars['grib']['reanalysis_pressure_levels']\
                   ['all_vars']:
            cols.append(Column(var, Float, nullable=False))

        return cols

    def get_era5_table_names(self, vars_mode):
        table_names = []
        # Get TC table and count its row number
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        # Loop all row of TC table
        for row in self.session.query(TCTable).filter(
            TCTable.date_time >= self.period[0],
            TCTable.date_time <= self.period[1]).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            # Get TC datetime
            tc_datetime = row.date_time

            # Get hit result and range of ERA5 data matrix near
            # TC center
            hit, lat1, lat2, lon1, lon2 = \
                    utils.get_subset_range_of_grib(
                        row.lat, row.lon, self.lat_grid_points,
                        self.lon_grid_points, self.edge, vars_mode='era5',
                        spatial_resolution=self.spa_resolu)
            if not hit:
                continue

            dirs = ['nw', 'sw', 'se', 'ne']
            r34 = dict()
            r34['nw'], r34['sw'], r34['se'], r34['ne'] = \
                    row.r34_nw, row.r34_sw, row.r34_se, row.r34_ne
            skip_compare = False
            for dir in dirs:
                if r34[dir] is None:
                    skip_compare = True
                    break
            if skip_compare:
                continue

            # Get name, sqlalchemy Table class and python original class
            # of ERA5 table
            table_name, sa_table, ERA5Table = self.get_era5_table_class(
                vars_mode, row.sid, tc_datetime)

            table_names.append(table_name)

        return table_names

    def get_era5_table_class(self, vars_mode, sid, dt):
        """Get table of ERA5 reanalysis.

        """
        dt_str = dt.strftime('%Y_%m%d_%H%M')
        table_name = f'era5_tc_{vars_mode}_{sid}_{dt_str}'

        class ERA5(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(ERA5, t)

            return table_name, None, ERA5

        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))

        if vars_mode == 'threeD':
            cols.append(Column('z', Integer, nullable=False))

        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))

        if vars_mode == 'threeD':
            cols.append(Column('pres_lvl', Integer, nullable=False))
            cols.append(Column('x_y_z', String(20), nullable=False,
                               unique=True))
            for var in self.threeD_vars:
                cols.append(Column(var, Float))

        elif vars_mode == 'surface_wind':
            cols.append(Column('x_y', String(20), nullable=False,
                               unique=True))
            for var in self.wind_vars:
                cols.append(Column(var, Float))

        elif vars_mode == 'surface_all_vars':
            cols.append(Column('x_y', String(20), nullable=False,
                               unique=True))
            for var in self.all_vars:
                if var == 'vorticity':
                    var = 'vorticity_relative'
                cols.append(Column(var, Float))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        mapper(ERA5, t)

        return table_name, t, ERA5

    def download_single_levels_vars(self, vars_mode, target_datetime,
                                    time_mode, times, area,
                                    match_satel, filename_suffix,
                                    show_info=False):
        era5_dirs = self.CONFIG['era5']['dirs']\
                ['reanalysis_single_levels']

        if vars_mode == 'tc':
            file_dir = (
                f"""{era5_dirs["tc"]}{match_satel}/"""
                f"""Y{target_datetime.year}/"""
                f"""M{str(target_datetime.month).zfill(2)}/"""
            )
            filename = target_datetime.strftime('%Y_%m%d')
            for t in times:
                filename = f'{filename}_{str(t).zfill(2)}'
            for a in area:
                filename = f'{filename}_{a}'
            filename = f'{filename}_{filename_suffix}.grib'
            file_path = f'{file_dir}{filename}'
            variables = self.vars['request']\
                    ['reanalysis_single_levels']['tc']
            vars_name = 'TC-relevant single levels variables'
        elif vars_mode == 'surface_wind':
            file_dir = (
                f"""{era5_dirs["10_metre_equivalent_neutral_wind"]}"""
                f"""{match_satel}/"""
                f"""Y{target_datetime.year}/"""
                f"""M{str(target_datetime.month).zfill(2)}/"""
            )
            filename = target_datetime.strftime('%Y_%m%d')
            for t in times:
                filename = f'{filename}_{str(t).zfill(2)}'
            for a in area:
                filename = f'{filename}_{a}'
            filename = f'{filename}_{filename_suffix}.grib'
            file_path = f'{file_dir}{filename}'
            variables = self.vars['request']\
                    ['reanalysis_single_levels']\
                    ['10_metre_equivalent_neutral_wind']
            vars_name = '10 metre equivalent neutral wind'
        else:
            return None

        if os.path.exists(file_path):
            return file_path

        if show_info:
            self.logger.info((f"""Downloading {vars_name} of {times} """
                              f"""hours on {target_datetime.date()} """
                              f"""from ERA5"""))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if time_mode == 'whole_day':
            hour_times = [f'{str(x).zfill(2)}:00' for x in range(24)]
        else:
            hour_times = [f'{str(x).zfill(2)}:00' for x in times]

        # ATTENTION: if area crosses the prime merdian, e.g.
        # area = [10, 358, 8, 2], ERA5 API cannot retrieve
        # requested data correctly
        request = {
            'product_type':'reanalysis',
            'format':'grib',
            'variable': variables,
            'year':f'{target_datetime.year}',
            'month':str(target_datetime.month).zfill(2),
            'day':str(target_datetime.day).zfill(2),
            'area': area,
            'time':hour_times
        }

        try:
            self.cdsapi_client.retrieve(
                'reanalysis-era5-single-levels',
                request,
                file_path)
        except Exception as msg:
            breakpoint()

        return file_path

    def download_pressure_levels_vars(self, vars_mode,
                                      target_datetime, time_mode,
                                      times, area, pressure_levels,
                                      match_satel, filename_suffix,
                                      show_info=False):
        era5_dirs = self.CONFIG['era5']['dirs']\
                ['reanalysis_pressure_levels']

        if vars_mode == 'tc':
            file_dir = (
                f"""{era5_dirs["tc"]}{match_satel}/"""
                f"""Y{target_datetime.year}/"""
                f"""M{str(target_datetime.month).zfill(2)}/"""
            )
            filename = target_datetime.strftime('%Y_%m%d')
            for t in times:
                filename = f'{filename}_{str(t).zfill(2)}'
            for a in area:
                filename = f'{filename}_{a}'
            filename = f'{filename}_{filename_suffix}.grib'
            file_path = f'{file_dir}{filename}'
            variables = self.vars['request']\
                    ['reanalysis_pressure_levels']['all_vars']
            vars_name = 'all pressure levels variables'
        else:
            return None

        if os.path.exists(file_path):
            return file_path

        if show_info:
            self.logger.info((f"""Downloading {vars_name} of {times} """
                              f"""hours on {target_datetime.date()} """
                              f"""from ERA5"""))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if time_mode == 'whole_day':
            hour_times = [f'{str(x).zfill(2)}:00' for x in range(24)]
        else:
            hour_times = [f'{str(x).zfill(2)}:00' for x in times]

        pres_lvls = [str(x) for x in pressure_levels]

        request = {
            'product_type':'reanalysis',
            'format':'grib',
            'pressure_level':pres_lvls,
            'variable': variables,
            'year':f'{target_datetime.year}',
            'month':str(target_datetime.month).zfill(2),
            'day':str(target_datetime.day).zfill(2),
            'area':area,
            'time':hour_times
        }

        try:
            self.cdsapi_client.retrieve(
                'reanalysis-era5-pressure-levels',
                request,
                file_path)
        except Exception as msg:
            breakpoint()

        return file_path

    def download_major(self, vars_mode, file_path, year, month):
        """Download major of ERA5 data which consists of main hour
        (0, 6, 12, 18 h) data in one month.

        """
        if os.path.exists(file_path):
            return

        request = {
            'product_type':'reanalysis',
            'format':'grib',
            'year':f'{year}',
            'month':str(month).zfill(2),
            'day':list(self.dt_major[year][month]['day']),
            'time':list(self.dt_major[year][month]['time'])
        }

        if vars_mode == 'threeD':
            request['variable'] = self.threeD_vars
            request['pressure_level'] = self.threeD_pres_lvl
        elif vars_mode == 'surface_wind':
            request['variable'] = self.wind_vars
            request['pressure_level'] = self.surface_pres_lvl
        elif vars_mode == 'surface_all_vars':
            request['variable'] = self.all_vars
            request['pressure_level'] = self.surface_pres_lvl

        self.cdsapi_client.retrieve(
            'reanalysis-era5-pressure-levels',
            request,
            file_path)

    def download_minor(self, vars_mode, file_path, year, month, day_str):
        """Download minor of ERA5 data which consists of hours except
        (0, 6, 12, 18 h) data in one day.

        """
        if os.path.exists(file_path):
            return

        request = {
            'product_type':'reanalysis',
            'format':'grib',
            'year':f'{year}',
            'month':str(month).zfill(2),
            'day':day_str,
            'time':list(self.dt_minor[year][month][day_str])
        }

        if vars_mode == 'threeD':
            request['variable'] = self.threeD_vars
            request['pressure_level'] = self.threeD_pres_lvl
        elif vars_mode == 'surface_wind':
            request['variable'] = self.wind_vars
            request['pressure_level'] = self.surface_pres_lvl
        elif vars_mode == 'surface_all_vars':
            request['variable'] = self.all_vars
            request['pressure_level'] = self.surface_pres_lvl

        self.cdsapi_client.retrieve(
            'reanalysis-era5-pressure-levels',
            request,
            file_path)

    def download_and_read(self, work_mode, vars_mode):
        """Download and read ERA5 data.

        """
        if work_mode == 'tc':
            self.download_and_read_tc_oriented(vars_mode)
        elif work_mode == 'scs':
            self.download_and_read_scs_oriented(vars_mode)

    def download_and_read_scs_oriented(self, vars_mode):
        self.logger.info((f"""SCS-orientedly downloading and """
                          f"""reading ERA5 reanalysis"""))
        delta = self.period[1] - self.period[0]

        if delta.seconds > 0:
            days_num = delta.days + 1
        else:
            days_num = delta.days

        for day in range(days_num):
            dt_cursor = self.period[0] + datetime.timedelta(days=day)

            self.logger.info((f"""Downloading ERA5 data on """
                              f"""{dt_cursor.strftime('%Y-%m-%d')}"""))
            file_path = self.download_surface_vars_of_whole_day(
                 vars_mode, dt_cursor)

            self.logger.info((f"""Reading SCS ERA5 data on """
                              f"""{dt_cursor.strftime('%Y-%m-%d')}"""))
            self.read('scs', vars_mode, file_path, dt_cursor)
            if self.save_disk:
                os.remove(file_path)

    def download_and_read_tc_oriented(self, vars_mode):
        self.logger.info((f"""TC-orientedly downloading and reading """
                          f"""ERA5 reanalysis: {vars_mode} mode"""))
        self._get_target_datetime()

        if vars_mode == 'threeD':
            major_dir = self.CONFIG['era5']['dirs']['threeD']['major']
            minor_dir = self.CONFIG['era5']['dirs']['threeD']['minor']
        elif vars_mode == 'surface_wind':
            major_dir = self.CONFIG['era5']['dirs']['surface_wind']\
                    ['major']
            minor_dir = self.CONFIG['era5']['dirs']['surface_wind']\
                    ['minor']
        elif vars_mode == 'surface_all_vars':
            major_dir = self.CONFIG['era5']['dirs']['surface_all_vars']\
                    ['major']
            minor_dir = self.CONFIG['era5']['dirs']['surface_all_vars']\
                    ['minor']

        os.makedirs(major_dir, exist_ok=True)
        os.makedirs(minor_dir, exist_ok=True)

        # Download and read major of ERA5
        for year in self.dt_major.keys():
            for month in self.dt_major[year].keys():
                file_path = (f'{major_dir}{year}'
                             + f'{str(month).zfill(2)}.grib')
                self.logger.info(f'Downloading major {file_path}')

                if not os.path.exists(file_path):
                    self.download_major(vars_mode, file_path, year,
                                        month)

                self.logger.info(f'Reading major {file_path}')
                self.read('tc', vars_mode, file_path)
                if self.save_disk:
                    os.remove(file_path)

        # Download and read minor of ERA5
        for year in self.dt_minor.keys():
            for month in self.dt_minor[year].keys():
                for day_str in self.dt_minor[year][month].keys():
                    file_path = (f'{minor_dir}{year}'
                                 + f'{str(month).zfill(2)}'
                                 + f'{day_str}.grib')
                    self.logger.info(f'Downloading minor {file_path}')

                    if not os.path.exists(file_path):
                        self.download_minor(vars_mode, file_path, year, month,
                                            day_str)

                    self.logger.info(f'Reading minor {file_path}')
                    self.read('tc', vars_mode, file_path)
                    if self.save_disk:
                        os.remove(file_path)

    def _get_radii_from_tc_row(self, tc_row):
        r34 = dict()
        r34['nw'], r34['sw'], r34['se'], r34['ne'] = \
                tc_row.r34_nw, tc_row.r34_sw, tc_row.r34_se, tc_row.r34_ne

        r50 = dict()
        r50['nw'], r50['sw'], r50['se'], r50['ne'] = \
                tc_row.r50_nw, tc_row.r50_sw, tc_row.r50_se, tc_row.r50_ne

        r64 = dict()
        r64['nw'], r64['sw'], r64['se'], r64['ne'] = \
                tc_row.r64_nw, tc_row.r64_sw, tc_row.r64_se, tc_row.r64_ne

        radii = {34: r34, 50: r50, 64: r64}

        return radii

    def _get_compare_latlon(self, surface):
        lons = set()
        lats = set()
        for pt in surface:
            lons.add(pt.lon)
            lats.add(pt.lat)
        lons = sorted(list(lons))
        lats = sorted(list(lats))

        return lats, lons

    def _draw_ibtracs_radii(self, ax, center, radii):
        radii_color = {34: 'yellow', 50: 'orange', 64: 'red'}
        dirs = ['ne', 'se', 'sw', 'nw']
        ibtracs_area = []

        for r in self.wind_radii:
            area_in_radii = 0
            for idx, dir in enumerate(dirs):
                if radii[r][dir] is None:
                    continue

                ax.add_patch(
                    mpatches.Wedge(
                        center,
                        r=radii[r][dir]*DEGREE_OF_ONE_NMILE,
                        theta1=idx*90, theta2=(idx+1)*90,
                        zorder=self.zorders['wedge'],
                        color=radii_color[r], alpha=0.6)
                )

                radii_in_km = radii[r][dir] * KM_OF_ONE_NMILE
                area_in_radii += math.pi * (radii_in_km)**2 / 4

            ibtracs_area.append(area_in_radii)

        return ibtracs_area

    def _draw_compare_basemap(self, ax, lon1, lon2, lat1, lat2):
        map = Basemap(llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2,
                      urcrnrlat=lat2, ax=ax)
        map.drawcoastlines(linewidth=3.0,
                           zorder=self.zorders['coastlines'])
        map.drawmapboundary(zorder=self.zorders['mapboundary'])
        # draw parallels and meridians.
        # label parallels on right and top
        # meridians on bottom and left
        parallels = np.arange(int(lat1), int(lat2), 2)
        # labels = [left,right,top,bottom]
        map.drawparallels(parallels,labels=[False,True,True,False])
        meridians = np.arange(int(lon1), int(lon2), 2)
        map.drawmeridians(meridians,labels=[True,False,False,True])

    def _get_compare_era5_windspd(self, surface, lats, lons):
        windspd = np.ndarray(shape=(len(lats), len(lons)),
                             dtype=float)
        for pt in surface:
            lon_idx = lons.index(pt.lon)
            lat_idx = lats.index(pt.lat)

            windspd[lat_idx][lon_idx] = 1.94384 * math.sqrt(
                pt.u_component_of_wind**2 + pt.v_component_of_wind**2)

        return windspd

    def _draw_era5_windspd(self, ax, lats, lons, windspd):
        # Plot windspd in knots with matplotlib's contour
        X, Y = np.meshgrid(lons, lats)
        Z = windspd

        windspd_levels = [5*x for x in range(15)]

        cs = ax.contour(X, Y, Z, levels=windspd_levels,
                        zorder=self.zorders['contour'], colors='k')
        ax.clabel(cs, inline=1, colors='k', fontsize=10)
        ax.contourf(X, Y, Z, levels=windspd_levels,
                    zorder=self.zorders['contourf'],
                    cmap=plt.cm.rainbow)

    def _get_era5_area(self, ax, lats, lons, windspd):
        X, Y = np.meshgrid(lons, lats)
        Z = windspd

        cs = ax.contour(X, Y, Z, levels=self.wind_radii)
        era5_area = []
        for i in range(len(self.wind_radii)):
            if windspd.max() < self.wind_radii[i]:
                era5_area.append(0)
                continue

            contour = cs.collections[i]
            paths = contour.get_paths()

            if not len(paths):
                continue

            vs = paths[0].vertices
            # Compute area enclosed by vertices.
            era5_area.append(abs(
                utils.area_of_contour(vs) * (KM_OF_ONE_DEGREE)**2))

        return era5_area

    def _set_basemap_title(self, ax, tc_row):
        title_prefix = (f'IBTrACS wind radii and ERA5 ocean surface wind '
                        + f'speed of'
                        + f'\n{tc_row.sid}')
        if tc_row.name is not None:
            tc_name =  f'({tc_row.name}) '
        title_suffix = f'on {tc_row.date_time}'
        ax.set_title(f'{title_prefix} {tc_name} {title_suffix}')

    def _set_bar_title_and_so_on(self, ax, tc_row, labels, x):
        title_prefix = (f'Area within wind radii of IBTrACS '
                        + f'and area within corresponding contour of ERA5'
                        + f'\n of {tc_row.sid}')
        if tc_row.name is not None:
            tc_name =  f'({tc_row.name}) '
        title_suffix = f'on {tc_row.date_time}'

        ax.set_title(f'{title_prefix} {tc_name} {title_suffix}')
        ax.set_ylabel('Area')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    def _draw_compare_area_bar(self, ax, ibtracs_area, era5_area, tc_row):
        labels = ['R34 area', 'R50 area', 'R64 area']
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        rects1 = ax.bar(x - width/2, ibtracs_area, width,
                         label='IBTrACS')
        rects2 = ax.bar(x + width/2, era5_area, width, label='ERA5')

        self._set_bar_title_and_so_on(ax, tc_row, labels, x)

        utils.autolabel(ax, rects1)
        utils.autolabel(ax, rects2)

    def compare_ibtracs_era5(self, vars_mode, ibtracs_table_row,
                             ERA5Table, draw, draw_map, draw_bar):
        tc_row = ibtracs_table_row
        lon_converted = tc_row.lon + 360 if tc_row.lon < 0 else tc_row.lon
        radii = self._get_radii_from_tc_row(tc_row)

        if vars_mode == 'threeD':
            surface = self.session.query(ERA5Table).filter(
                ERA5Table.z == 11)
        elif vars_mode == 'surface_wind' or vars_mode == 'surface_all_vars':
            surface = self.session.query(ERA5Table)

        lats, lons = self._get_compare_latlon(surface)
        lon1, lon2 = min(lons), max(lons)
        lat1, lat2 = min(lats), max(lats)

        fig = plt.figure()
        if draw_map and draw_bar:
            fig.set_size_inches(25, 10)
            map_subplot = 121
            bar_subplot = 122
        else:
            fig.set_size_inches(10, 10)
            map_subplot = bar_subplot = 111

        # Draw ax1 to compare IBTrACS wind radii with ERA5 wind speed contour
        ax1 = fig.add_subplot(map_subplot, aspect='equal')
        ax1.axis([lon1, lon2, lat1, lat2])

        self._draw_compare_basemap(ax1, lon1, lon2, lat1, lat2)
        self._set_basemap_title(ax1, tc_row)

        center = (lon_converted, tc_row.lat)
        windspd = self._get_compare_era5_windspd(surface, lats, lons)
        self._draw_era5_windspd(ax1, lats, lons, windspd)

        ibtracs_area = self._draw_ibtracs_radii(ax1, center, radii)
        era5_area = self._get_era5_area(ax1, lats, lons, windspd)

        self.write_area_compare(tc_row, ibtracs_area, era5_area)

        if not draw:
            plt.close(fig)
            return

        if not draw_map:
            ax1.remove()

        # Draw ax2 to compare area within IBTrACS wind radii with
        # corresponding area of ERA5 wind speed contour
        ax2 = fig.add_subplot(bar_subplot)
        self._draw_compare_area_bar(ax2, ibtracs_area, era5_area, tc_row)

        if not draw_bar:
            ax2.remove()

        fig.tight_layout()
        fig_dir = self.CONFIG["result"]["dirs"]["fig"]\
                ['ibtracs_vs_era5']
        fig_path = (f'{fig_dir}'
                    + f'era5_vs_ibtracs_{tc_row.sid}_'
                    + f'{tc_row.name}_{tc_row.date_time}_'
                    + f'{lon_converted}_{tc_row.lat}.png')
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    def write_area_compare(self, tc_row, ibtracs_area, era5_area):
        area = {
            'ibtracs': ibtracs_area,
            'era5': era5_area
        }

        CompareTable = self.create_area_compare_table()
        row = CompareTable()
        # Write area and metrics into row
        row.sid = tc_row.sid
        row.date_time = tc_row.date_time

        for type in ['ibtracs', 'era5']:
            for idx, r in enumerate(self.wind_radii):
                setattr(row, f'{type}_r{r}_area', float(area[type][idx]))
        row.sid_date_time = f'{tc_row.sid}_{tc_row.date_time}'

        utils.bulk_insert_avoid_duplicate_unique(
            [row], self.CONFIG['database']\
            ['batch_size']['insert'],
            CompareTable, ['sid_date_time'], self.session,
            check_self=True)

    def create_area_compare_table(self):
        """Get table of ERA5 reanalysis.

        """
        table_name = f'wind_radii_area_compare'

        class WindRadiiAreaCompare(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(WindRadiiAreaCompare, t)

            return WindRadiiAreaCompare

        cols = []
        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('sid', String(13), nullable=False))
        cols.append(Column('date_time', DateTime, nullable=False))
        for type in ['ibtracs', 'era5', 'smap']:
            for r in self.wind_radii:
                col_name = f'{type}_r{r}_area'
                cols.append(Column(col_name, Float, nullable=False))
        cols.append(Column('sid_date_time', String(50), nullable=False,
                           unique=True))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        mapper(WindRadiiAreaCompare, t)

        metadata.create_all()
        self.session.commit()

        return WindRadiiAreaCompare

    def read(self, work_mode, vars_mode, file_path, dt_cursor):
        if work_mode == 'tc':
            self.read_tc_oriented(vars_mode, file_path)
        elif work_mode == 'scs':
            self.read_scs_oriented(vars_mode, file_path, dt_cursor)

    def read_scs_oriented(self, vars_mode, file_path, dt_cursor):
        # Open ERA5 grib data and read 6-hourly
        grbidx = pygrib.index(file_path, 'dataTime')

        for hourtime in range(0, 2400,
                              self.CONFIG['product']\
                              ['temporal_resolution']):

            # Create ERA5 table for SCS
            SCSERA5 = self.create_scs_era5_table(dt_cursor, hourtime)

            selected_grbs = grbidx.select(dataTime=hourtime)
            # Generate frame of one 6-hour SCS ERA5 table
            table_entity = self.gen_scs_era5_entity(SCSERA5)

            total = len(selected_grbs)

            for idx, grb in enumerate(selected_grbs):
                info = (f"""\rReading grbs on hour """
                        f"""{int(hourtime/100)} {idx+1}/{total}""")
                print(f'\r{info}', end='')
                # Traverse all data point in ERA5 grib message,
                # find corresponding row in SCS ERA5 table frame
                # and fill its environmental variables
                table_entity = self.fill_scs_era5_table_entity(
                    grb, table_entity)

                # Temporarily not interpolate the space between
                # ERA5 0.25 degree grid points

                # Method_1: conventional interpolation methods

                # Method_2: GAN

            # Insert entity into database
            utils.bulk_insert_avoid_duplicate_unique(
                table_entity, self.CONFIG['database']\
                ['batch_size']['insert'],
                SCSERA5, ['x_y'], self.session,
                check_self=True)
            utils.delete_last_lines()
        print('Done')

    def fill_scs_era5_table_entity(self, grb, table_entity):
        data, lats, lons = grb.data(-90, 90, 0, 360)
        data = np.flip(data, 0)

        # Generate name which is the same with table column
        name = grb.name.replace(" ", "_").lower()
        if name == 'vorticity_(relative)':
            name = 'vorticity_relative'

        for row in table_entity:
            if (row.lat not in self.grid_pts['era5']['lat']
                or row.lon not in self.grid_pts['era5']['lon']):
                continue
            lat_idx = self.grid_pts['era5']['lat'].index(row.lat)
            lon_idx = self.grid_pts['era5']['lon'].index(row.lon)
            setattr(row, name, float(data[lat_idx][lon_idx]))

        return table_entity

    def gen_scs_era5_entity(self, SCSERA5):
        entity = []
        lons_num = int((self.lon2 - self.lon1) / self.spa_resolu) + 1
        lats_num = int((self.lat2 - self.lat1) / self.spa_resolu) + 1
        # Traverse through meridians first
        for lon in np.linspace(self.lon1, self.lon2, lons_num):
            x = self.grid_lons.index(lon)

            # Traverse through parallels then
            for lat in np.linspace(self.lat1, self.lat2, lats_num):
                is_land = bool(globe.is_land(lat, lon))
                if is_land:
                    continue

                y = self.grid_lats.index(lat)

                row = SCSERA5()
                row.x = x
                row.y = y
                row.lon = float(lon)
                row.lat = float(lat)
                row.land = is_land
                row.x_y = f'{x}_{y}'
                row.satel_era5_diff_mins = 0

                entity.append(row)

        return entity

    def create_scs_era5_table(self, dt_cursor, hourtime):
        table_name = utils.gen_scs_era5_table_name(dt_cursor, hourtime)

        class SCSERA5(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(SCSERA5, t)

            return SCSERA5

        cols = []

        cols.append(Column('key', Integer, primary_key=True))
        cols.append(Column('x', Integer, nullable=False))
        cols.append(Column('y', Integer, nullable=False))
        cols.append(Column('lon', Float, nullable=False))
        cols.append(Column('lat', Float, nullable=False))
        cols.append(Column('land', Boolean, nullable=False))
        cols.append(Column('x_y', String(30), nullable=False,
                           unique=True))

        # cols.append(Column('grid_pt_key', Integer, primary_key=True))

        era5_cols = self.get_era5_columns()

        cols = cols + era5_cols

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(SCSERA5, t)

        self.session.commit()

        return SCSERA5

    def read_tc_oriented(self, vars_mode, file_path):
        # load grib file
        grbs = pygrib.open(file_path)

        # Get TC table and count its row number
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        tc_query = self.session.query(TCTable)
        total = tc_query.count()
        del tc_query
        count = 0
        info = f'Reading reanalysis data of TC records'
        self.logger.info(info)

        # Loop all row of TC table
        for row in self.session.query(TCTable).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            # Get TC datetime
            tc_datetime = row.date_time

            # Get hit result and range of ERA5 data matrix near
            # TC center
            hit, lat1, lat2, lon1, lon2 = \
                    utils.get_subset_range_of_grib(
                        row.lat, row.lon, self.lat_grid_points,
                        self.lon_grid_points, self.edge, vars_mode='era5',
                        spatial_resolution=self.spa_resolu)
            if not hit:
                continue

            count += 1
            print(f'\r{info} {count}/{total}', end='')

            dirs = ['nw', 'sw', 'se', 'ne']
            r34 = dict()
            r34['nw'], r34['sw'], r34['se'], r34['ne'] = \
                    row.r34_nw, row.r34_sw, row.r34_se, row.r34_ne
            skip_compare = False
            for dir in dirs:
                if r34[dir] is None:
                    skip_compare = True
                    break
            if skip_compare:
                continue

            # Get name, sqlalchemy Table class and python original class
            # of ERA5 table
            table_name, sa_table, ERA5Table = self.get_era5_table_class(
                vars_mode, row.sid, tc_datetime)

            # Create entity of ERA5 table
            era5_table_entity = self._gen_whole_era5_table_entity(
                vars_mode, ERA5Table, lat1, lat2, lon1, lon2)

            # Record number of successfully reading data matrix of ERA5
            # grib file near TC center
            read_hit_count = 0

            # Loop all messages of grib file which consists of
            # all variables in all pressure levels
            for m in range(grbs.messages):
                grb = grbs.message(m+1)

                # Generate datetime of message and compare it with TC's
                grb_date, grb_time = str(grb.dataDate), str(grb.dataTime)
                if grb_time == '0':
                    grb_time = '000'
                grb_datetime = datetime.datetime.strptime(
                    f'{grb_date}{grb_time}', '%Y%m%d%H%M%S')
                if tc_datetime != grb_datetime:
                    continue

                # extract corresponding data matrix in ERA5 reanalysis
                read_hit = self._read_grb_matrix(vars_mode, era5_table_entity,
                                                 grb, lat1, lat2, lon1,
                                                 lon2)
                if read_hit:
                    read_hit_count += 1

            # Skip this turn of loop if not getting data matrix
            if not read_hit_count:
                continue

            # When ERA5 table doesn't exists, sa_table is None.
            # So need to create it.
            if sa_table is not None:
                # Create table of ERA5 data cube
                sa_table.create(self.engine)
                self.session.commit()

            # Write extracted data matrix into DB
            start = time.process_time()
            if vars_mode == 'threeD':
                utils.bulk_insert_avoid_duplicate_unique(
                    era5_table_entity,
                    int(self.CONFIG['database']['batch_size']['insert']/10),
                    ERA5Table, ['x_y_z'], self.session,
                    check_self=True)
            elif vars_mode == 'surface_wind' or vars_mode == 'surface_all_vars':
                utils.bulk_insert_avoid_duplicate_unique(
                    era5_table_entity,
                    int(self.CONFIG['database']['batch_size']['insert']/10),
                    ERA5Table, ['x_y'], self.session,
                    check_self=True)
            end = time.process_time()

            self.logger.debug((f'Bulk inserting ERA5 data into '
                               + f'{table_name} in {end-start:2f} s'))

            self.compare_ibtracs_era5(vars_mode, row, ERA5Table, draw=True,
                                      draw_map=True, draw_bar=False)
        utils.delete_last_lines()
        print('Done')

    def _gen_whole_era5_table_entity(self, vars_mode, ERA5Table,
                                     lat1, lat2, lon1, lon2):
        """Generate entity of ERA5 table. It represents a threeD grid of
        which center of bottom is the closest grid point near TC
        center.

        """
        entity = []
        half_edge_indices = (self.edge / 2 / self.spa_resolu)

        if vars_mode == 'threeD':
            for x in range(self.threeD_grid['lon_axis']):
                for y in range(self.threeD_grid['lat_axis']):
                    for z in range(self.threeD_grid['height']):
                        pt = ERA5Table()
                        pt.x = x - half_edge_indices
                        pt.y = y - half_edge_indices
                        pt.z = z
                        pt.x_y_z = f'{pt.x}_{pt.y}_{pt.z}'

                        pt.lat = lat1 + (y+0.5) * self.spa_resolu
                        pt.lon = (lon1 + (x+0.5) * self.spa_resolu) % 360
                        pt.pres_lvl = int(self.threeD_pres_lvl[z])

                        entity.append(pt)

        elif vars_mode == 'surface_wind' or vars_mode == 'surface_all_vars':
            for x in range(self.threeD_grid['lon_axis']):
                for y in range(self.threeD_grid['lat_axis']):
                    pt = ERA5Table()
                    pt.x = x - half_edge_indices
                    pt.y = y - half_edge_indices
                    pt.x_y = f'{pt.x}_{pt.y}'

                    pt.lat = lat1 + (y+0.5) * self.spa_resolu
                    pt.lon = (lon1 + (x+0.5) * self.spa_resolu) % 360

                    entity.append(pt)

        return entity

    def _read_grb_matrix(self, vars_mode, era5, grb, lat1, lat2, lon1, lon2):
        """Read data matrix of ERA5 of particular variable in particular
        pressure level.

        """
        data, lats, lons = grb.data(lat1, lat2, lon1, lon2)

        # Shape of data is (lat_grid_num, lon_grid_num).
        # In the extracted subset of ERA5 data,
        # lats[0] is latitude of the northest horizontal line,
        # lons[0] is longitude of the northest horizontal line.
        # So need to flip the data matrix along latitude axis
        # to be the same of RSS satellite data matrix arrangement.
        data = np.flip(data, 0)
        lats = np.flip(lats, 0)
        lons = np.flip(lons, 0)
        # After fliping, data[0][0] is the data of smallest latitude
        # and smallest longitude

        name = grb.name.replace(" ", "_").lower()
        if name == 'vorticity_(relative)':
            name = 'vorticity_relative'
        if vars_mode == 'threeD':
            z = self.threeD_pres_lvl.index(str(grb.level))
        hit_count = 0

        for x in range(self.threeD_grid['lon_axis']):
            for y in range(self.threeD_grid['lat_axis']):
                # ERA5 grid starts from (lat=-90, lon=0),
                # while RSS grid starts from (lat=-89.875, lon=0.125).
                # So ERA5 grid points is the corners of RSS grid cells.
                # Since we decide to use RSS grid,
                # to get the value of a cell, have to get its 4 corners
                # first: ul=upper_left, ll=lower_left, lr=lower_right,
                # ur=upper_right
                """
                value_ll = utils.convert_dtype(data[y][x])
                value_ul = utils.convert_dtype(data[y+1][x])
                value_lr = utils.convert_dtype(data[y][x+1])
                value_ur = utils.convert_dtype(data[y+1][x+1])
                values = []
                for var in [value_ul, value_ll, value_lr, value_ur]:
                    if var is not None:
                        values.append(var)

                if not len(values):
                    continue

                # Average four corner's data value
                value = sum(values)/float(len(values))
                """
                corners = []
                for tmp_y in [y, y+1]:
                    for tmp_x in [x, x+1]:
                        corners.append(data[tmp_y][tmp_x])

                if not len(corners):
                    continue

                value = float( sum(corners) / len(corners) )

                hit_count += 1
                if vars_mode == 'threeD':
                    index = ((x * self.threeD_grid['lat_axis'] * \
                              self.threeD_grid['height'])
                             + y * self.threeD_grid['height']
                             + z)
                elif vars_mode == 'surface_wind' or vars_mode == 'surface_all_vars':
                    index = x * self.threeD_grid['lat_axis'] + y

                setattr(era5[index], name, value)

        if hit_count:
            return True
        else:
            return False

    def _update_major_datetime_dict(self, dt_dict, year, month, day, hour):
        """Update major datetime dictionary.

        """
        day_str = str(day).zfill(2)
        time_str = f'{str(hour).zfill(2)}:00'

        if year not in dt_dict:
            dt_dict[year] = dict()
        if month not in dt_dict[year]:
            dt_dict[year][month] = dict()
            dt_dict[year][month]['day'] = set()
            dt_dict[year][month]['time'] = set()

        dt_dict[year][month]['day'].add(day_str)
        dt_dict[year][month]['time'].add(time_str)

    def _update_minor_datetime_dict(self, dt_dict, year, month, day, hour):
        """Update minor datetime dictionary.

        """
        day_str = str(day).zfill(2)
        time_str = f'{str(hour).zfill(2)}:00'

        if year not in dt_dict:
            dt_dict[year] = dict()
        if month not in dt_dict[year]:
            dt_dict[year][month] = dict()
        if day_str not in dt_dict[year][month]:
            dt_dict[year][month][day_str] = set()

        dt_dict[year][month][day_str].add(time_str)

    def _get_target_datetime(self):
        """Get major datetime dictionary and minor datetime
        dictionary.

        """
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        dt_major = dict()
        dt_minor = dict()

        for row in self.session.query(TCTable).filter(
            TCTable.date_time >= self.period[0],
            TCTable.date_time <= self.period[1]).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            dirs = ['nw', 'sw', 'se', 'ne']
            r34 = dict()
            r34['nw'], r34['sw'], r34['se'], r34['ne'] = \
                    row.r34_nw, row.r34_sw, row.r34_se, row.r34_ne
            skip_compare = False
            for dir in dirs:
                if r34[dir] is None:
                    skip_compare = True
                    break
            if skip_compare:
                continue

            year, month = row.date_time.year, row.date_time.month
            day, hour = row.date_time.day, row.date_time.hour
            if hour in self.main_hours:
                self._update_major_datetime_dict(dt_major, year,
                                                 month, day, hour)
            else:
                self._update_minor_datetime_dict(dt_minor, year,
                                                 month, day, hour)

        self.dt_major = dt_major
        self.dt_minor = dt_minor
