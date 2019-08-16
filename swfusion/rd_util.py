# !/usr/bin/env python

import math
import sys
from functools import partial

from ascat_daily import ASCATDaily
from quikscat_daily_v4 import QuikScatDaily
from windsat_daily_v7 import WindSatDaily

def read_daily_satel(satel_name, file_path, missing_val=-999.0):
    if satel_name == 'ascat':
        dataset = ASCATDaily(file_path, missing=missing_val)
    elif satel_name == 'qscat':
        dataset = QuikScatDaily(file_path, missing=missing_val)
    elif satel_name == 'wsat':
        dataset = WindSatDaily(file_path, missing=missing_val)
    else:
        sys.exit('Invalid satellite name')

    if not dataset.variables:
        sys.exit('Filenot found')

    return dataset

def show_dimensions(ds):
    print('')
    print('Dimensions')
    for dim in ds.dimensions:
        aline = ' '.join([' '*3, dim, ':', str(ds.dimensions[dim])])
        print(aline)

def show_variables(ds):
    print('')
    print('Variables:')
    for var in ds.variables:
        aline = ' '.join([' '*3, var, ':', ds.variables[var].long_name])
        print(aline)

def show_validrange(ds):
    print('')
    print('Valid min and max and units:')
    for var in ds.variables:
        aline = ' '.join([' '*3, var, ':',
                str(ds.variables[var].valid_min), 'to',
                str(ds.variables[var].valid_max),
                '(',ds.variables[var].units,')'])
        print(aline)

def find_index(range, lat_or_lon):
    # latitude: from -89.875 to 89.875, 720 values, interval = 0.25
    # longitude: from 0.125 to 359.875, 1440 values, interval = 0.25
    res = []
    for idx, val in enumerate(range):
        if lat_or_lon == 'lat':
            delta = val + 89.875
        elif lat_or_lon == 'lon':
            delta = val - 0.125
        else:
            print('Error parameter lat_or_lon: ' + lat_or_lon)
            exit(0)
        intervals = delta / 0.25
        if not idx:
            res.append(math.ceil(intervals))
        else:
            res.append(math.floor(intervals))

    return res

def cut_map(satel_name, dataset, region, year, month, day,
            missing_val=-999.0):
    min_lat, max_lat = find_index([region[0], region[1]], 'lat')
    lat_indices = [x for x in range(min_lat, max_lat+1)]
    min_lon, max_lon = find_index([region[2], regionn[3]], 'lon')
    lon_indices = [x for x in range(min_lon, max_lon+1)]

    data_list = []
    iasc = [0, 1]
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    num = len(lat_indices) * len(lon_indices) * 2
    num_c1, num_c2, num_c3 = 0, 0, 0
    vars = dataset.variables
    for j in lat_indices:
        for k in lon_indices:
            for i in iasc:
                cut_missing = vars['nodata'][i][j][k]
                cut_mingmt = vars['mingmt'][i][j][k]
                cut_land = vars['land'][i][j][k]
                if satel_name == 'ascat' or satel_name == 'qscat':
                    cut_wspd = vars['windspd'][i][j][k]
                    cut_wdir = vars['winddir'][i][j][k]
                    cut_rain = vars['scatflag'][i][j][k]
                elif satel_name == 'wsat':
                    cut_rain = vars['rain'][i][j][k]
                    cut_wspd_lf = vars['w-lf'][i][j][k]
                    cut_wspd_mf = vars['w-mf'][i][j][k]
                    cut_wspd_aw = vars['w-aw'][i][j][k]
                    cut_wdir = vars['wdir'][i][j][k]
                else:
                    sys.exit('satel_name is wrong.')

                if cut_missing or cut_land or cut_wdir == missing_val:
                    # same pass condition for all satellites
                    num_c1 += 1
                    continue

                if satel_name == 'ascat' or satel_name == 'qscat':
                    if cut_wspd == missing_val:
                        num_c2 += 1
                        continue
                elif satel_name == 'wsat':
                    if (cut_wspd_lf == missing_val
                        or cut_wspd_mf == missing_val
                        or cut_wspd_aw == missing_val):
                        # at least one of three wind speed is missing
                        num_c3 += 1
                        continue

                data_point = {}
                data_point['iasc'] = i
                data_point['lat'] = vars['latitude'][j]
                data_point['lon'] = vars['longitude'][k]
                data_point['wdir'] = cut_wdir
                data_point['rain'] = cut_rain
                data_point['time'] = cut_mingmt
                data_point['year'] = year
                data_point['month'] = month
                data_point['day'] = day

                if satel_name == 'ascat' or satel_name == 'qscat':
                    data_point['wspd'] = cut_wspd
                elif satel_name == 'wsat':
                    data_point['w-lf'] = cut_wspd_lf
                    data_point['w-mf'] = cut_wspd_mf
                    data_point['w-aw'] = cut_wspd_aw

                data_list.append(data_point)

   #  print()
   #  print('total data point: ' + str(num))
   #  print('skip condition 1: ' + str(num_c1))
   #  print('skip condition 2: ' + str(num_c2))
   #  print('skip condition 3: ' + str(num_c3))
   #  print('returned data point: ' + str(len(data_list)))
   #  print()

    return data_list

def narrow_map(dataset, region):
    # Find rectangle range of area
    min_lat, max_lat = find_index([region[0], region[1]], 'lat')
    min_lon, max_lon = find_index([region[2], region[3]], 'lon')

    map = {}
    map['wspd'] = []
    map['wdir'] = []
    map['rain'] = []
    map['time'] = []
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    iasc = [0, 1]
    vars = dataset.variables
    for i in iasc:
        wspd = vars['windspd'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        wdir = vars['winddir'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        rain = vars['scatflag'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        time = vars['mingmt'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        map['wspd'].append(wspd)
        map['wdir'].append(wdir)
        map['rain'].append(rain)
        map['time'].append(time)

    return map
