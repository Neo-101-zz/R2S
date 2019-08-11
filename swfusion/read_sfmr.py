# !/usr/bin/env python

from datetime import date
from datetime import timedelta
import os
import pickle

import conf_sfmr
import netcdf_util

def main():
    confs = conf_sfmr.configure()
    with open(confs['var_dir'] + 'hurr_year.pkl', 'rb') as fr:
        hurr_year = pickle.load(fr)
    data_root_dir = confs['hurr_dir']
    all_file_suffix = []
    for hurr in hurr_year.keys():
        year = hurr_year[hurr]
        print('\nProcessing hurricane ' + hurr + ' in ' + year + '\n')
        data_spec_dir = data_root_dir + year + '/' + hurr + '/'
        files = os.listdir(data_spec_dir)
        for file in files:
            if not file.endswith('.nc'):
                files.remove(file)
        files.sort()

        # Test if there exists files with same suffix
        file_suffix = [f[-13:-3] for f in files]
        all_file_suffix = all_file_suffix + file_suffix
        l = all_file_suffix
        duplicates = set([x for x in l if l.count(x) > 1])
        if len(duplicates):
            print('Find duplicated suffix of files:')
            print(duplicates)
            exit(0)

        os.makedirs(confs['pickle_dir'], exist_ok=True)

        for file in files:
            pickle_name = file[-13:-3] + '_' + hurr + '.pkl'
            pickle_path = confs['pickle_dir'] + pickle_name
            if os.path.exists(pickle_path):
                continue
            print('Processing ' + pickle_name)
            data = netcdf_util.ReadNetcdf(data_spec_dir + file)
            wspd = []
            lat = []
            lon = []
            time = []
            precise_time = []
            # FLAG: validity flag, 0 if data is valid
            for i in range(len(data.dataset.variables['FLAG'])):
                if data.dataset.variables['FLAG'][i] == 0:
                    # SWS: SFMR wind speed
                    wspd.append(data.dataset.variables['SWS'][i])
                    # FDIR: Flt. lvl. wind direction
                    lat.append(data.dataset.variables['LAT'][i])
                    l = data.dataset.variables['LON'][i]
                    if l < 0:
                        l += 360
                    lon.append(l)
                    t = str(int(data.dataset.variables['TIME'][i]))
                    precise_time.append(t)
                    s = int(t[-2:])
                    min = t[-4:-2]
                    # Is it possible that minute string is empty ?
                    if min == '':
                        m = 0
                    else:
                        m = int(min)
                    hour = t[-6:-4]
                    # Is it possible that hourstring is empty ?
                    if hour == '':
                        h = 0
                    else:
                        h = int(hour)
                    # Round seconds to minutes
                    if s >= 30:
                        m += 1
                    time.append(h * 60 + m)
                    # print(time)
            pickle_file = open(pickle_path, 'wb')
            pickle.dump({'wspd': wspd,
                         'lat': lat,
                         'lon': lon,
                         'time': time,
                         'precise_time': precise_time},
                        pickle_file)
            pickle_file.close()

if __name__ == '__main__':
    main()
