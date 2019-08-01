# !/usr/bin/env python

import os

def configure():
    confs = {}

    confs['url_base'] = 'http://www.ndbc.noaa.gov/data/historical/cwind/'
    confs['station_page'] = 'http://www.ndbc.noaa.gov/station_page.php'
    confs['var_dir'] = '../variable/'
    confs['station_dir'] = '../station_info/'
    confs['cwind_dir'] = '../cwind_data/'
    confs['station_csv'] = '../stations.csv'
    confs['pickle_dir'] = '../cwind_pickle/'
    confs['retry_times'] = 10
    confs['data_name_len'] = 16

    confs['input_year_prompt'] = '\nInputting target year(s)' \
        + '\n\nSupporting 3 input modes:' \
        + '\n\n1. Single input, e.g. "1997 1998 2000"' \
        + '\n2. Range input, e,g. "1997-2001 2005-2008"' \
        + '\n3. Hybrid input, e.g. "1997 1999-2001 2012 2003-2010"' \
        + '\n\nInput target year(s) (only press ENTER to skip): '

    confs['input_station_prompt'] = '\nInputting target station(s)' \
            + '\n\nOnly support single input mode, ' \
            + 'e.g. "41001 41004 lonf1"' \
            + '\n\nInput target station(s) (only press ENTER to skip): '
    confs['input_min_latitude_prompt'] = 'Inputting MINIMAL latitude ' \
            + '(negative if in southern hemisphere, ' \
            + 'press ENTER to input default value -90): '
    confs['default_min_latitude'] = -90
    confs['input_max_latitude_prompt'] = 'Inputting MAXIMAL latitude ' \
            + '(negative if in southern hemisphere, ' \
            + 'press ENTER to input default value 90): '
    confs['default_max_latitude'] = 90
    confs['input_min_longitude_prompt'] = 'Enter MINIMAL longitude ' \
            + '(negative if in western hemisphere, '\
            + 'press ENTER to input default value 0): '
    confs['default_min_longitude'] = 0
    confs['input_max_longitude_prompt'] = 'Enter MAXIMAL longitude ' \
            + '(negative if in western hemisphere, ' \
            + 'press ENTER to input default value 360): '
    confs['default_max_longitude'] = 360
    return confs
