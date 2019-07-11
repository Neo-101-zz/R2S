# !/usr/bin/env python

import os

def configure():
    confs = {}
    confs['url_base'] = 'http://www.ndbc.noaa.gov/data/historical/cwind/'
    confs['station_page'] = 'http://www.ndbc.noaa.gov/station_page.php'
    confs['var_dir'] = '../variable/'
    confs['station_dir'] = '../station_info/'
    confs['cwind_dir'] = '../cwind_data/'
    confs['retry_times'] = 10
    confs['data_name_len'] = 16
    return confs
