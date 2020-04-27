import datetime
import logging

from sqlalchemy.ext.declarative import declarative_base
from netCDF4 import Dataset

Base = declarative_base()

class CenterOfTC(object):

    def __init__(self, date_time, lon, lat):
        self.date_time = date_time
        self.lon = lon
        self.lat = lat

class WindSpeedCell(object):

    def __init__(self, date_time, lon, lat, windspd):
        self.date_time = date_time
        self.lon = lon
        self.lat = lat
        self.windspd = windspd

class SmartComparer(object):

    def __init__(self, CONFIG, period, basin):
        self.CONFIG = CONFIG
        self.period = period
        self.basin = basin
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)
        utils.setup_database(self, Base)

    def matchup_smap_sfmr(self):
        """Match SMAP and SFMR data around TC.

        """
        center_datetime = dict()
        center_lonlat = dict()

        # Get table class of sfmr brief info
        SFMRInfo = utils.get_class_by_tablename(
            self.engine,
            self.CONFIG['sfmr']['table_names']['brief_info'])

        sfmr_info_query = self.session.query(SFMRInfo).filter(
            SFMRInfo.start_datetime < self.period[1],
            SFMRInfo.end_datetime > self.period[0])

        # Traverse SFMR files
        for sfmr_info in sfmr_info_query:
            tc_name = sfmr_info.hurr_name
            sfmr_path = (f"""self.CONFIG['sfmr']['dir']['hurr']"""
                         f"""{sfmr_info.start_datetime.year}"""
                         f"""/{tc_name}/{sfmr_info.file_path}""")

            # SFMR track was closest to TC center
            # when SFMR SWS reached its peak
            center_datetime['sfmr'] = self.time_of_sfmr_peak_wind(
                sfmr_path)

            # Find where was TC center when SFMR SWS reached its peak
            center_lonlat['sfmr'] = self.lonlat_of_tc_center(
                tc_name, center_datetime['sfmr'])

            # "TC center of SFMR" means "TC center when SFMR SWS reached
            # its peak".  "TC center of SMAP" means "TC center when and
            # where SMAP is enough close to SFMR track".

            # Farthest permitted spatial distance between
            # "TC center of SFMR" and "TC center of SMAP"
            # max_center_spatial_dist = 

            # "region of center cells" within circle area with radius of
            # "max_center_spat_dist" around "TC center of SFMR"
            center_cells = self.cells_around_tc_center(
                center_lonlat['sfmr'], max_center_dist)

            # Farthest permitted temporal distance between 
            # "TC center of SFMR" and "TC center of SMAP"
            # max_center_temporal_dist = 

            # Check the existence of SMAP data in "region of center
            # cells" within temporal window
            exist, center_datetime['smap'], center_lonlat['smap'] = \
                    self.cover_tc_center(center_cells,
                                         center_datetime['sfmr'],
                                         max_center_temporal_dist)
            if not exist:
                continue

            # Extract lon, lat and wind speed of SMAP
            smap_pts = self.extract_smap(
                center_datetime['smap'], center_lonlat['smap'])

            # Largest permitted change in intensity
            # max_intensity_change = 

            # To avoid cases where TC had changed too much, we need to
            # estimate the change in intensity between SMAP and SFMR
            intensity_change = self.intensity_change_between_shift(
                tc_name, center_datetime)

            if intensity_change > max_intensity_change:
                continue

            # Study region around TC center
            # square_edge = 

            # Resample SFMR SWS
            sfmr_track, resampled_sfmr_pts = self.resample_sfmr(
                sfmr_path, center_datetime['sfmr'], center_lonlat['sfmr'])

            # Calculate shift of SFMR
            shift = self.cal_shift(center_lonlat)

            # Shift SFMR track and resampled SFMR SWS
            sfmr_track, resampled_sfmr_pts = self.do_shift(
                sfmr_track, resampled_sfmr_pts)

            self.record_matchup(sfmr_track, resampled_sfmr_pts)

    def time_of_sfmr_peak_wind(self, sfmr_path):
        """Get the datetime when SFMR wind reached its peak value.

        """
        dataset = Dataset(sfmr_path)
        vars = dataset.variables

        breakpoint()
