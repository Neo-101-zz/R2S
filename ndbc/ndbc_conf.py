# !/usr/bin/env python

import os

def configure():
    confs = {}
    confs['var_dir'] = './variable/'
    confs['station_dir'] = './station_info/'
    confs['cwind_dir'] = './cwind_data/'
    confs['retry_times'] = 10
    return confs
