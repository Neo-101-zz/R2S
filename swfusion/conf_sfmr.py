# !/usr/bin/env python

import os

def configure():
    confs = {}

    sfmr_data_dir = '../data/sfmr/'
    confs['hurricane_data_url'] = \
            'https://www.aoml.noaa.gov/hrd/data_sub/hurr.html'
    confs['storm_SFMR_url_prefix'] = \
            'https://www.aoml.noaa.gov/hrd/Storm_pages/'
    confs['storm_SFMR_url_suffix'] = '/sfmr.html'
    confs['var_dir'] = sfmr_data_dir + 'variable/'
    confs['hurr_dir'] = sfmr_data_dir + 'hurr_data/'
    confs['pickle_dir'] = sfmr_data_dir + 'sfmr_pickle/'
    confs['data_name_len'] = 25

    confs['input_prompt'] = '\nInputting target hurricane(s)' \
        + '\n\nFormats: ' \
        + '[year_1, hurricane_name_1] ' \
        + '[year_2, hurricane_name_2] ' \
        + '...' \
        + '\n\nInput target hurricane(s): '
    return confs
