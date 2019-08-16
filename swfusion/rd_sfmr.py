# !/usr/bin/env python

from datetime import date
from datetime import timedelta
import os
import pickle

import load_config
import netcdf_util

def dump_sfmr_netcdf(file_path, pickle_path):
    """Dump one SFMR NetCDF file into one pickle file.

    """
    if not file_path.endswith('.nc'):
        return
    if os.path.exists(pickle_path):
        return
    print('Processing ' + pickle_path.split('/')[-1])
    data = netcdf_util.ReadNetcdf(file_path)
    wspd, lat, lon = [], [], []
    # minute-unit time and precise (second-unit) time
    mins_time, precise_time = [], []
    vars = data.dataset.variables
    # FLAG: validity flag, 0 if data is valid
    for i in range(len(vars['FLAG'])):
        if vars['FLAG'][i] == 0:
            # SWS: SFMR wind speed
            wspd.append(vars['SWS'][i])
            # FDIR: Flt. lvl. wind direction
            lat.append(vars['LAT'][i])
            lon.append((vars['LON'][i] + 360) % 360)
            time_ = str(int(vars['TIME'][i]))
            precise_time.append(time_)
            sec = int(time_[-2:])
            # Is it possible that minute string is empty ?
            min = int(time_[-4:-2]) if time_[-4:-2] else 0
            # Is it possible that hourstring is empty ?
            hour = int(time_[-6:-4]) if time_[-6:-4] else 0
            # Round seconds to minutes
            if sec >= 30:
                min += 1
            mins_time.append(hour * 60 + min)

    pickle_file = open(pickle_path, 'wb')
    pickle.dump({'wspd': wspd, 'lat': lat, 'lon': lon,
                 'time': mins_time, 'precise_time': precise_time},
                pickle_file)
    pickle_file.close()

def read_sfmr(CONFIG):
    """Read SFMR NetCDF files into pickle files.

    """
    with open(CONFIG['vars_path']['sfmr']['year_hurr'], 'rb') as fr:
        year_hurr = pickle.load(fr)
    data_root_dir = CONFIG['dirs']['sfmr']['hurr']

    for year in year_hurr.keys():
        for hurr in year_hurr[year]:
            print('\nProcessing hurricane {0} in {1}\n'.format(
                hurr, year))
            spec_data_dir = '{0}{1}/{2}/'.format(
                data_root_dir, year, hurr)
            files = os.listdir(spec_data_dir)

            pickle_dir = CONFIG['dirs']['sfmr']['pickle']
            os.makedirs(pickle_dir, exist_ok=True)

            for file in files:
                file_path = spec_data_dir + file
                pickle_name = file[-13:-3] + '_' + hurr + '.pkl'
                pickle_path = pickle_dir + pickle_name
                dump_sfmr_netcdf(file_path, pickle_path)

if __name__ == '__main__':
    CONFIG = load_config.load_config()
    read_sfmr(CONFIG)
