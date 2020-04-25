import datetime
import logging
import time

from global_land_mask import globe
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import mapper
from netCDF4 import Dataset
import numpy as np
import pygrib
from scipy import interpolate

import utils
import satel_scs
import era5

Base = declarative_base()

class matchManager(object):

    def __init__(self, CONFIG, period, region, basin, passwd, save_disk):
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.save_disk = save_disk
        self.engine = None
        self.session = None
        self.basin = basin

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

        self.years = [x for x in range(self.period[0].year,
                                       self.period[1].year+1)]
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.lats = dict()
        self.lons = dict()
        self.spa_resolu = dict()

        self.spa_resolu['smap'] = self.CONFIG['rss'][
            'spatial_resolution']
        self.lats['smap'] = [
            y * self.spa_resolu['smap'] - 89.875 for y in range(720)]
        self.lons['smap'] = [
            x * self.spa_resolu['smap'] + 0.125 for x in range(1440)]

        self.spa_resolu['era5'] = self.CONFIG['era5'][
            'spatial_resolution']
        self.lats['era5'] = [
            y * self.spa_resolu['era5'] - 90 for y in range(721)]
        self.lons['era5'] = [
            x * self.spa_resolu['era5'] for x in range(1440)]

        self.edge = self.CONFIG['regression']['edge_in_degree']
        self.half_edge = self.edge / 2
        self.half_edge_grid_intervals = int(
            self.half_edge / self.spa_resolu['smap'])

        self.pres_lvls = self.CONFIG['era5']['pres_lvls']

        utils.setup_database(self, Base)

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.sources = ['era5', 'smap']

        self.extract()

    def extract(self):
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
        # Traverse WP TCs
        for idx, tc in enumerate(tc_query):
            try:
                converted_lon = utils.longtitude_converter(
                    tc.lon, '360', '-180')
                if bool(globe.is_land(tc.lat, converted_lon)):
                    continue
                if tc.date_time.minute or tc.date_time.second:
                    continue
                if idx < total - 1:
                    next_tc = tc_query[idx + 1]
                    # This TC and next TC is same TC
                    if tc.sid == next_tc.sid:
                        self.extract_between_two_tc_records(tc, next_tc)
                    # This TC differents next TC
                    else:
                        self.extract_detail(tc.sid, tc.date_time,
                                            tc.lon, tc.lat)
                else:
                    self.extract_detail(tc.sid, tc.date_time, tc.lon,
                                        tc.lat)
            except Exception as msg:
                breakpoint()
                exit(msg)

    def extract_between_two_tc_records(self, tc, next_tc):
        """
        Notes
        -----
        In this project, the range of longitude is from 0 to 360, so there
        may be data leap near the prime merdian.  And the interpolation
        should be ajusted.

        """
        # Temporal shift
        delta = next_tc.date_time - tc.date_time
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are far away in time
        if delta.days:
            self.extract_detail(tc.sid, tc.date_time, tc.lon, tc.lat)
            return
        hours = int(delta.seconds / 3600)
        # Skip interpolating between two TC recors if two neighbouring
        # records of TC are too close in time
        if not hours:
            self.extract_detail(tc.sid, tc.date_time, tc.lon, tc.lat)
            return
        # Spatial shift
        lon_shift, lat_shift = utils.get_center_shift_of_two_tcs(
            next_tc, tc)
        hourly_lon_shift = lon_shift / hours
        hourly_lat_shift = lat_shift / hours
        # Extract from the interval between two TC records
        for h in range(hours):
            interp_dt = tc.date_time + datetime.timedelta(
                seconds = h * 3600)
            interp_tc_lon = (h * hourly_lon_shift + tc.lon + 360) % 360
            interp_tc_lat = (h * hourly_lat_shift + tc.lat)
            self.extract_detail(tc.sid, interp_dt, interp_tc_lon,
                                interp_tc_lat)

    def extract_detail(self, tc_id, tc_dt, tc_lon, tc_lat):
        SMAPERA5 = self.create_smap_era5_table(tc_dt)
        # Skip this turn if there is no SMAP data around TC
        try:
            data, hourtimes, area = self.extract_smap(
                tc_id, tc_dt, tc_lon, tc_lat, SMAPERA5)
            if not len(data):
                return
        except Exception as msg:
            breakpoint()
            exit(msg)

        try:
            data = self.extract_era5(tc_id, tc_dt, data, hourtimes, area)
        except Exception as msg:
            breakpoint()
            exit(msg)

        utils.bulk_insert_avoid_duplicate_unique(
            data, self.CONFIG['database']\
            ['batch_size']['insert'],
            SMAPERA5, ['satel_datetime_lon_lat'], self.session,
            check_self=True)

    def get_square_around_tc(self, tc_lon, tc_lat):
        """Get indices of square corners around tropical cyclone center
        in grid.

        parameters
        ----------
        tc_lon: float
        tc_lat: float

        Return
        ------
        success: bool
        lat1_idx: int
        lat2_idx: int
        lon1_idx: int
        lon2_idx: int
        lat1: float
        lon1: float

        """
        try:
            tc_lon_in_grid, tc_lon_in_grid_idx = \
                    utils.get_nearest_element_and_index(
                        self.lons['smap'], tc_lon)
            tc_lat_in_grid, tc_lat_in_grid_idx = \
                    utils.get_nearest_element_and_index(
                        self.lats['smap'], tc_lat)
            lat1_idx = tc_lat_in_grid_idx - self.half_edge_grid_intervals
            lat1 = self.lats['smap'][lat1_idx]
            lat2_idx = tc_lat_in_grid_idx + self.half_edge_grid_intervals
            lat2 = self.lats['smap'][lat2_idx]

            if lat1 < -90 or lat2 > 90:
                success = False
            else:
                success = True

            lons_num = len(self.lons['smap'])
            lon1_idx = tc_lon_in_grid_idx - self.half_edge_grid_intervals
            lon1_idx = (lon1_idx + lons_num) % lons_num
            lon1 = self.lons['smap'][lon1_idx]
            lon2_idx = tc_lon_in_grid_idx + self.half_edge_grid_intervals
            lon2_idx = (lon2_idx + lons_num) % lons_num
            lon2 = self.lons['smap'][lon2_idx]

            # ATTENTION: if area crosses the prime merdian, e.g.
            # area = [10, 358, 8, 2], ERA5 API cannot retrieve
            # requested data correctly,  so skip this situation
            if lon1_idx >= lon2_idx:
                success = False

        except Exception as msg:
            breakpoint()
            exit(msg)

        return success, lat1_idx, lat2_idx, lon1_idx, lon2_idx,\
                lat1, lon1

    def extract_smap(self, tc_id, tc_dt, tc_lon, tc_lat, SMAPERA5):
        """Extract SMAP data according to tropical cyclone data
        from IBTrACS.

        parameters
        ----------
        tc_dt: datetime
            Datetime of tropical cyclone.  May be interpolated value,
            not originally from IBTrACS.
        tc_lon: float
            Longitude of tropical cyclone.  May be interpolated value,
            not originally from IBTrACS.
        tc_lat: float
            Latitude of tropical cyclone.  May be interpolated value,
            not originally from IBTrACS.
        SMAPERA5: table class
            Repersentation of SMAP data and matching ERA5 data around
            tropical cyclone.

        Return
        ------
        data: list
            List of extracted SMAP data.  Elements are SMAPERA5 type.
        hourtimes: list
            List of hour time that SMAP data is closest to.  e.g. 2
            o'clock and 5 o'clock.

        """
        self.logger.info((f"""Extracting SMAP data on {tc_dt} """
                          f"""around TC {tc_id}"""))
        try:
            satel_manager = satel_scs.SCSSatelManager(
                self.CONFIG, self.period, self.region,
                self.db_root_passwd,
                save_disk=self.save_disk, work=False)
            smap_file_path = satel_manager.download('smap', tc_dt)
            if smap_file_path is None:
                return [], None, None
            data, hourtimes, area = self.get_smap_part(
                SMAPERA5, tc_id, tc_dt, tc_lon, tc_lat, smap_file_path)
        except Exception as msg:
            breakpoint()
            exit(msg)

        return data, hourtimes, area

    def get_smap_part(self, SMAPERA5, tc_id, tc_dt, tc_lon, tc_lat,
                      smap_file_path):
        success, lat1_idx, lat2_idx, lon1_idx, lon2_idx, lat1, lon1 = \
                self.get_square_around_tc(tc_lon, tc_lat)
        if not success:
            return [], None, None

        dataset = Dataset(smap_file_path)
        # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which
        # faster than 1 m/s, so must disable auto mask
        dataset.set_auto_mask(False)
        vars = dataset.variables
        # Square around TC does not cross the prime meridian
        minute = vars['minute'][lat1_idx:lat2_idx+1,
                                lon1_idx:lon2_idx+1, :]
        wind = vars['wind'][lat1_idx:lat2_idx+1,
                            lon1_idx:lon2_idx+1, :]

        lats_num, lons_num, passes_num = minute.shape
        minute_missing = self.CONFIG['smap']['missing_value']['minute']
        wind_missing = self.CONFIG['smap']['missing_value']['wind']

        data = []
        lats = []
        lons = []
        north = -90
        west = 360
        south = 90
        east = 0
        hourtimes = set()

        for y in range(lats_num):
            lat_of_row = y * self.spa_resolu['smap'] + lat1

            for x in range(lons_num):
                lon_of_col = x * self.spa_resolu['smap'] + lon1
                lon_of_col = (lon_of_col + 360) % 360

                for i in range(passes_num):
                    try:
                        if (minute[y][x][i] == minute_missing
                            or wind[y][x][i] == wind_missing):
                            continue
                        if minute[y][x][0] == minute[y][x][1]:
                            continue
                        if minute[y][x][i] == 1440:
                            continue
                        time_ = datetime.time(
                            *divmod(int(minute[y][x][i]), 60), 0)
                        # Temporal window is one hour
                        pixel_dt = datetime.datetime.combine(
                            tc_dt.date(), time_)
                        # MUST use abs() method
                        delta = abs(pixel_dt - tc_dt)
                        if delta.days or delta.seconds > 1800:
                            continue

                        # SMAP originally has land mask, so it's not
                        # necessary to check whether each pixel is land
                        # or ocean
                        row = SMAPERA5()
                        row.sid = tc_id
                        row.satel_datetime = pixel_dt
                        row.x = x - self.half_edge_grid_intervals
                        row.y = y - self.half_edge_grid_intervals

                        row.lon = lon_of_col
                        # if row.lon < west:
                        #     west = row.lon
                        # if row.lon > east:
                        #     east = row.lon
                        lons.append(row.lon)

                        row.lat = lat_of_row
                        # if row.lat < south:
                        #     south = row.lat
                        # if row.lat > north:
                        #     north = row.lat
                        lats.append(row.lat)

                        row.satel_datetime_lon_lat = (
                            f"""{row.satel_datetime}"""
                            f"""_{row.lon}_{row.lat}""")
                        row.smap_windspd = float(wind[y][x][i])

                        this_hourtime = utils.hour_rounder(
                            row.satel_datetime).hour
                        # Skip situation that hour is rounded to next day
                        if (row.satel_datetime.hour == 23
                            and this_hourtime == 0):
                            continue

                        # Strictest reading rule: None of columns is none
                        skip = False
                        for key in row.__dict__.keys():
                            if getattr(row, key) is None:
                                skip = True
                                break
                        if skip:
                            continue
                        else:
                            data.append(row)
                            hourtimes.add(this_hourtime)
                    except Exception as msg:
                        breakpoint()
                        exit(msg)

        if not len(data):
            return data, None, None

        north = max(lats)
        south = min(lats)
        east = max(lons)
        west = min(lons)
        # 'area' parameter to request ERA5 data via API:
        # North, West, South, East
        # e.g. [12.125, 188.875, 3.125, 197.875]
        #
        # Due to difference bewteen ERA5 and RSS grid, need
        # to expand the area a little to request an ERA5 grid
        # consists of all minimum squares around all RSS points.
        #
        # Considering the spatial resolution of ERA5 ocean waves
        # are 0.5 degree x 0.5 degree, not 0.25 x 0.25 degree of 
        # atmosphere, we need to expand a little more, at least 0.5
        # degree
        diff = 0.5
        area = [north + diff, (west - diff + 360) % 360,
                south - diff, (east + diff + 360) % 360]
        # --------------------------------------------------
        # Updated 2019/12/24
        #
        # It seems that ERA5 will autonomically extract grid according
        # to user's input of 'area' parameter when requesting ERA5
        # data via API.
        #
        # For example, if 'area' parameter is
        # [1.125, 0.625, 0.125, 1.625], the ERA5 grib file's lats will
        # be [0.125, 0.375, 0.625, 0.875, 1.125] and its lons will be
        # [0.625, 0.875, 1.125. 1.375, 1.625], which is 0.25 degree
        # grid but the coordinates are not the multiple of 0.25.
        #
        # Becaues we consider ERA5 reanalysis grid coordinates are the
        # multiple of 0.25 before, so we will squeeze the expanded area
        # a little to a range which corners' coordiantes are the
        # multiple of 0.25.
        #
        # North, West, South, East
        for idx, val in enumerate(area):
            if utils.is_multiple_of(val, 0.125):
                # decrease north and east a little
                if not idx or idx == 3:
                    area[idx] = val - 0.125
                # increase west and south a little
                else:
                    area[idx] = val + 0.125
        area[1] = (area[1] + 360) % 360
        area[3] = (area[3] + 360) % 360

        return data, list(hourtimes), area

    def extract_era5(self, tc_id, tc_dt, satel_part, hourtimes, area):
        """Extract ERA5 data to match satellite data.

        parameters
        ----------
        tc: TC table
            An instance of TC table.
        satel_part: SatelERA5
            An instance of SatelERA5 which has been partly filled by
            satellite data.

        Return
        ------
        era5_step_2: SatelERA5
            An instance of SatelERA5 which has been fully filled by
            satellite data and matching ERA5 data.
        """
        self.logger.info((f"""Extracting ERA5 data on {tc_dt.date()} """
                          f"""hour {hourtimes} within {area} around """
                          f"""TC {tc_id}"""))
        try:
            era5_step_1, pres_lvls = self.extract_era5_single_levels(
                tc_id, tc_dt, satel_part, hourtimes, area)

            era5_step_2 = self.extract_era5_pressure_levels(
                tc_id, tc_dt, era5_step_1, hourtimes, area, pres_lvls)
        except Exception as msg:
            breakpoint()
            exit(msg)

        return era5_step_2

    def extract_era5_single_levels(self, tc_id, tc_dt, satel_part,
                                   hourtimes, area):
        self.logger.info((f"""Extracting single levels reanalysis """
                          f"""of ERA5"""))
        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = \
                era5_manager.download_single_levels_vars(
                    'tc', tc_dt, '', hourtimes, area, 'smap', tc_id)

        try:
            era5_step_1, pres_lvls = self.add_era5_single_levels(
                era5_file_path, tc_dt, satel_part, hourtimes, area)
        except Exception as msg:
            breakpoint()

        return era5_step_1, pres_lvls

    def add_era5_single_levels(self, era5_file_path, tc_dt, satel_part,
                               hourtimes, area):
        hourtime_row_mapper = self.get_hourtime_row_mapper(satel_part)
        north, west, south, east = area

        grbs = pygrib.open(era5_file_path)
        messages_num = grbs.messages
        grbs.close()
        data_num = len(satel_part)
        total = data_num * messages_num
        count = 0

        grbidx = pygrib.index(era5_file_path, 'dataTime')
        indices_of_rows_to_delete = set()

        # For every hour, update corresponding rows with grbs
        for hourtime in range(0, 2400, 100):
            if not len(hourtime_row_mapper[hourtime]):
                continue
            grb_time = datetime.time(int(hourtime/100), 0, 0)

            selected_grbs = grbidx.select(dataTime=hourtime)

            for grb in selected_grbs:
                # Generate name which is the same with table column
                name = utils.process_grib_message_name(grb.name)
                grb_spa_resolu = grb.jDirectionIncrementInDegrees
                # data() method of pygrib is time-consuming
                # So apply it to global area then update all
                # smap part with grb of specific hourtime,
                # which using data() method as less as possible
                data, lats, lons = grb.data(south, north, west, east)
                data = np.flip(data, 0)
                lats = np.flip(lats, 0)
                lons = np.flip(lons, 0)

                masked_data = False
                # MUST check masked array like this, because if an array
                # is numpy.ma.core.MaskedArray, it is numpy.ndarray too.
                # So only directly check whether an array is instance
                # of numpy.ma.core.MaskedArray is safe.
                if isinstance(data, np.ma.core.MaskedArray):
                    masked_data = True

                # Update all rows which matching this hourtime
                for row_idx in hourtime_row_mapper[hourtime]:
                    count += 1
                    # print((f"""\r{name}: {count}/{total}"""), end='')
                    row = satel_part[row_idx]

                    row.era5_datetime = datetime.datetime.combine(
                        tc_dt.date(), grb_time)

                    satel_minute = (row.satel_datetime.hour * 60
                                     + row.satel_datetime.minute)
                    grb_minute = int(hourtime/100) * 60
                    row.satel_era5_diff_mins = \
                            satel_minute - grb_minute

                    try:
                        latlons, latlon_indices = \
                                utils.get_era5_corners_of_rss_cell(
                                    row.lat, row.lon, lats, lons,
                                    grb_spa_resolu)
                    except Exception as msg:
                        breakpoint()
                        exit(msg)
                    lat1, lat2, lon1, lon2 = latlons
                    lat1_idx, lat2_idx, lon1_idx, lon2_idx = \
                            latlon_indices

                    # Check out whether there is masked cell in square
                    if masked_data:
                        skip_row = False
                        for tmp_lat_idx in [lat1_idx, lat2_idx]:
                            for tmp_lon_idx in [lon1_idx, lon2_idx]:
                                if data.mask[tmp_lat_idx][tmp_lon_idx]:
                                    skip_row = True
                                    indices_of_rows_to_delete.add(
                                        row_idx)
                        if skip_row:
                            continue

                    square_data = data[lat1_idx:lat2_idx+1,
                                       lon1_idx:lon2_idx+1]
                    square_lats = lats[lat1_idx:lat2_idx+1,
                                       lon1_idx:lon2_idx+1]
                    square_lons = lons[lat1_idx:lat2_idx+1,
                                       lon1_idx:lon2_idx+1]

                    # ERA5 atmospheric variable
                    if grb_spa_resolu == 0.25:
                        value = float(square_data.mean())
                    # ERA5 oceanic variable
                    else:
                        value = self.value_of_rss_pt_in_era5_square(
                            square_data, square_lats, square_lons,
                            row.lat, row.lon)

                    setattr(row, name, value)

                utils.delete_last_lines()
                # print(f'{name}: Done')

        grbidx.close()

        # Move rows of satel_part which should not deleted to a new
        # list to accomplish filtering rows with masked data
        new_satel_part = []
        for idx, row in enumerate(satel_part):
            if idx not in indices_of_rows_to_delete:
                new_satel_part.append(row)

        pres_lvls = []
        for row in new_satel_part:
            nearest_pres_lvl, nearest_pres_lvl_idx = \
                    utils.get_nearest_element_and_index(
                        self.pres_lvls,
                        row.mean_sea_level_pressure / 100)

            windspd, winddir = utils.compose_wind(
                row.neutral_wind_at_10_m_u_component,
                row.neutral_wind_at_10_m_v_component,
                'o')
            row.era5_10m_neutral_equivalent_windspd = windspd
            row.era5_10m_neutral_equivalent_winddir = winddir

            row.smap_u_wind, row.smap_v_wind = utils.decompose_wind(
                row.smap_windspd, winddir, 'o')

            pres_lvls.append(nearest_pres_lvl)

        return new_satel_part, pres_lvls

    def value_of_rss_pt_in_era5_square(self, data, lats, lons,
                                       rss_lat, rss_lon):
        if lats.shape != (2, 2) or lons.shape != (2, 2):
            self.logger.error((f"""Not a square consists of four """
                               f"""ERA5 grid points"""))

        f = interpolate.interp2d(lons, lats, data)
        value = f(rss_lon, rss_lat)

        return float(value)

    def get_hourtime_row_mapper(self, satel_part):
        satel_day = satel_part[0].satel_datetime.day
        hourtime_row_mapper = dict()

        for hourtime in range(0, 2400, 100):
            hourtime_row_mapper[hourtime] = []

        for idx, row in enumerate(satel_part):
            hour_roundered_dt = utils.hour_rounder(row.satel_datetime)
            # Skip situation that rounded hour is on next day
            if hour_roundered_dt.day == satel_day:
                closest_time = 100 * hour_roundered_dt.hour
                hourtime_row_mapper[closest_time].append(idx)

        return hourtime_row_mapper

    def extract_era5_pressure_levels(self, tc_id, tc_dt, era5_step_1,
                                     hourtimes, area, pres_lvls):
        self.logger.info((f"""Extracting pressure levels reanalysis """
                          f"""of ERA5 on pressure levels of """
                          f"""{set(pres_lvls)}"""))

        era5_manager = era5.ERA5Manager(self.CONFIG, self.period,
                                        self.region,
                                        self.db_root_passwd,
                                        work=False,
                                        save_disk=self.save_disk,
                                        work_mode='',
                                        vars_mode='')
        era5_file_path = \
                era5_manager.download_pressure_levels_vars(
                    'tc', tc_dt, '', hourtimes, area,
                    sorted(list(set(pres_lvls))), 'smap', tc_id)

        era5_step_2 = self.add_era5_pressure_levels(
            era5_file_path, tc_dt, era5_step_1, hourtimes, area,
            pres_lvls)

        return era5_step_2

    def add_era5_pressure_levels(self, era5_file_path, tc_dt,
                                 era5_step_1, hourtimes, area,
                                 pres_lvls):
        hourtime_row_mapper = self.get_hourtime_row_mapper(era5_step_1)
        north, west, south, east = area

        grbs = pygrib.open(era5_file_path)
        messages_num = grbs.messages
        grbs.close()
        data_num = len(era5_step_1)
        total = data_num * messages_num
        count = 0

        grbidx = pygrib.index(era5_file_path, 'dataTime')
        indices_of_rows_to_delete = set()

        # For every hour, update corresponding rows with grbs
        for hourtime in range(0, 2400, 100):
            if not len(hourtime_row_mapper[hourtime]):
                continue
            grb_time = datetime.time(int(hourtime/100), 0, 0)

            selected_grbs = grbidx.select(dataTime=hourtime)

            for grb in selected_grbs:
                # Generate name which is the same with table column
                name = utils.process_grib_message_name(grb.name)
                grb_spa_resolu = grb.jDirectionIncrementInDegrees
                # data() method of pygrib is time-consuming
                # So apply it to global area then update all
                # smap part with grb of specific hourtime,
                # which using data() method as less as possible
                data, lats, lons = grb.data(south, north, west, east)
                data = np.flip(data, 0)
                lats = np.flip(lats, 0)
                lons = np.flip(lons, 0)

                masked_data = False
                # MUST check masked array like this, because if an array
                # is numpy.ma.core.MaskedArray, it is numpy.ndarray too.
                # So only directly check whether an array is instance
                # of numpy.ma.core.MaskedArray is safe.
                if isinstance(data, np.ma.core.MaskedArray):
                    masked_data = True

                # Update all rows which matching this hourtime
                for row_idx in hourtime_row_mapper[hourtime]:
                    count += 1
                    # print((f"""\r{name}: {count}/{total}"""), end='')

                    # Skip this turn if pressure level of grb does not
                    # equal to the pressure level of point of
                    # era5_step_1
                    if pres_lvls[row_idx] != grb.level:
                        continue

                    row = era5_step_1[row_idx]

                    era5_datetime = datetime.datetime.combine(
                        tc_dt.date(), grb_time)
                    if row.era5_datetime != era5_datetime:
                        self.logger.error((f"""datetime not same """
                                           f"""in two steps of """
                                           f"""extracting ERA5"""))

                    satel_minute = (row.satel_datetime.hour * 60
                                     + row.satel_datetime.minute)
                    grb_minute = int(hourtime/100) * 60
                    satel_era5_diff_mins = \
                            satel_minute - grb_minute
                    if row.satel_era5_diff_mins != satel_era5_diff_mins:
                        self.logger.error((f"""diff_mins not same """
                                           f"""in two steps of """
                                           f"""extracting ERA5"""))

                    latlons, latlon_indices = \
                            utils.get_era5_corners_of_rss_cell(
                                row.lat, row.lon, lats, lons,
                                grb_spa_resolu)
                    lat1, lat2, lon1, lon2 = latlons
                    lat1_idx, lat2_idx, lon1_idx, lon2_idx = \
                            latlon_indices

                    # Check out whether there is masked cell in square
                    if masked_data:
                        skip_row = False
                        for tmp_lat_idx in [lat1_idx, lat2_idx]:
                            for tmp_lon_idx in [lon1_idx, lon2_idx]:
                                if data.mask[tmp_lat_idx][tmp_lon_idx]:
                                    skip_row = True
                                    indices_of_rows_to_delete.add(
                                        row_idx)
                        if skip_row:
                            continue

                    square_data = data[lat1_idx:lat2_idx+1,
                                       lon1_idx:lon2_idx+1]
                    square_lats = lats[lat1_idx:lat2_idx+1,
                                       lon1_idx:lon2_idx+1]
                    square_lons = lons[lat1_idx:lat2_idx+1,
                                       lon1_idx:lon2_idx+1]

                    if grb_spa_resolu == 0.25:
                        value = float(data.mean())
                    else:
                        value = self.value_of_rss_pt_in_era5_square(
                            square_data, square_lats, square_lons,
                            row.lat, row.lon)

                    setattr(row, name, value)

                utils.delete_last_lines()
                # print(f'{name}: Done')

        grbidx.close()

        # Move rows of era5_step_1 which should not deleted to a new
        # list to accomplish filtering rows with masked data
        result = []
        for idx, row in enumerate(era5_step_1):
            if idx not in indices_of_rows_to_delete:
                result.append(row)

        return result

    def create_smap_era5_table(self, dt):
        table_name = utils.gen_tc_satel_era5_tablename('smap', dt,
                                                       self.basin)

        class Satel(object):
            pass

        if self.engine.dialect.has_table(self.engine, table_name):
            metadata = MetaData(bind=self.engine, reflect=True)
            t = metadata.tables[table_name]
            mapper(Satel, t)

            return Satel

        cols = utils.get_basic_satel_era5_columns(tc_info=True)

        cols.append(Column('smap_windspd', Float, nullable=False))

        cols.append(Column('smap_u_wind', Float, nullable=False))
        cols.append(Column('smap_v_wind', Float, nullable=False))

        era5_ = era5.ERA5Manager(self.CONFIG, self.period, self.region,
                                 self.db_root_passwd, False,
                                 self.save_disk, '', 'tc')
        era5_cols = era5_.get_era5_columns()
        cols = cols + era5_cols

        cols.append(Column('era5_10m_neutral_equivalent_windspd',
                           Float, nullable=False))
        cols.append(Column('era5_10m_neutral_equivalent_winddir',
                           Float, nullable=False))

        metadata = MetaData(bind=self.engine)
        t = Table(table_name, metadata, *cols)
        metadata.create_all()
        mapper(Satel, t)

        self.session.commit()

        return Satel
