# !/usr/bin/env python

import math
import sys

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

def cut_map(satel_name, dataset, lats, lons, month, day, missing_val):
    lat_indices = []
    lon_indices = []

    lat_bounds = find_index(lats, 'lat')
    lat_indices = [x for x in range(lat_bounds[0], lat_bounds[1]+1)]
    lon_bounds = find_index(lons, 'lon')
    lon_indices = [x for x in range(lon_bounds[0], lon_bounds[1]+1)]

    data_list = []
    iasc = [0, 1]
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    for j in lat_indices:
        for k in lon_indices:
            for i in iasc:
                cut_missing = dataset.variables['nodata'][i][j][k]
                cut_mingmt = dataset.variables['mingmt'][i][j][k]
                cut_land = dataset.variables['land'][i][j][k]
                if (satel_name == 'ascat'
                    or satel_name == 'qscat'):
                    cut_wspd = dataset.variables['windspd'][i][j][k]
                    cut_wdir = dataset.variables['winddir'][i][j][k]
                    cut_rain = dataset.variables['scatflag'][i][j][k]
                elif satel_name == 'windsat':
                    cut_rain = dataset.variables['rain'][i][j][k]
                    cut_wspd_lf = dataset.variables['w-lf'][i][j][k]
                    cut_wspd_mf = dataset.variables['w-mf'][i][j][k]
                    cut_wspd_aw = dataset.variables['w-aw'][i][j][k]
                    cut_wdir = dataset.variables['wdir'][i][j][k]
                else:
                    sys.exit('satel_name is wrong.')

                if (cut_missing
                    or cut_land
                    or cut_wdir == missing_val):
                    # same pass condition for all satellites
                    continue

                if (satel_name == 'ascat'
                    or satel_name == 'qscat'):
                    # pass condtion for ascat and qscat
                    if cut_wspd == missing_val:
                        continue
                elif satel_name == 'windsat':
                    if (cut_wspd_lf == missing_val
                        or cut_wspd_mf == missing_val
                        or cut_wspd_aw == missing_val):
                        # at least one of three wind speed is missing
                        continue

                data_point = {}
                data_point['iasc'] = i
                data_point['lat'] = dataset.variables['latitude'][j]
                data_point['lon'] = dataset.variables['longitude'][k]
                data_point['wdir'] = cut_wdir
                data_point['rain'] = cut_rain
                data_point['time'] = cut_mingmt
                data_point['month'] = month
                data_point['day'] = day

                if (satel_name == 'ascat'
                    or satel_name == 'qscat'):
                    data_point['wspd'] = cut_wspd
                elif satel_name == 'windsat':
                    data_point['w-lf'] = cut_wspd_lf
                    data_point['w-mf'] = cut_wspd_mf
                    data_point['w-aw'] = cut_wspd_aw

                data_list.append(data_point)

    return data_list
