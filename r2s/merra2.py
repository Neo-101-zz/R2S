import datetime
import logging
import math
import os
import sys
import time

import requests

class MERRA2Manager(object):

    def __init__(self, CONFIG, period, work):
        self.CONFIG = CONFIG
        self.period = period
        self.engine = None
        self.session = None

        self.logger = logging.getLogger(__name__)
        self.merra2_config = self.CONFIG['merra2']

        self.download_surface_wind(
            datetime.date(2017, 4, 16),
            [round(35.8-4, 3), round(309.7-360-4, 3),
             round(35.8+4, 3), round(309.7-360+4, 3)],
            '2017106N36310'
        )

    def download_surface_wind(self, target_date, area, filename_suffix):
        nc_name_majority = (
            f"""{self.merra2_config["filenames"]["prefix"]}"""
            f"""{target_date.strftime("%Y%m%d")}""")
        normal_nc_name = f'{nc_name_majority}.nc4'
        sub_nc_name = f'{nc_name_majority}.SUB.nc'

        south, west, north, east = area
        area_str = f'{south}%2C{west}%2C{north}%2C{east}'

        url = (
            f"""{self.merra2_config["urls"]["head"]}"""
            f"""%2F{target_date.strftime("%Y")}"""
            f"""%2F{target_date.strftime("%m")}"""
            f"""%2F{normal_nc_name}"""
            f"""&FORMAT=bmM0Lw"""
            f"""&BBOX={area_str}"""
            f"""&LABEL={sub_nc_name}"""
            f"""{self.merra2_config["urls"]["surface_wind_tail"]}"""
        )
        dir = (
            f"""{self.merra2_config['dirs']['surface_wind']}"""
            f"""Y{target_date.strftime("%Y")}/"""
            f"""M{target_date.strftime("%m")}/"""
        )
        filename = (
            f"""{target_date.strftime("%Y_%m%d")}"""
            f"""_{north}_{west}_{south}_{east}"""
            f"""_{filename_suffix}"""
            f""".nc"""
        )
        file_path = f'{dir}{filename}'

        if os.path.exists(file_path):
            return file_path

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        result = requests.get(url)
        try:
            result.raise_for_status()
            f = open(file_path, 'wb')
            f.write(result.content)
            f.close()
            print(f'contents of MERRA-2 written to {filename}')
        except:
            self.logger.error((
                f"""requests.get() returned an error code: """
                f"""result.status_code"""))
