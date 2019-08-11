import sys
import math
import time

from ascat_daily import ASCATDaily

def read_ascat(filename, missing_val):
    dataset = ASCATDaily(filename, missing=missing_val)
    if not dataset.variables:
        sys.exit('file not found')
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

def cut_map(dataset, lats, lons, month, day, missing_val):
    lat_indices = []
    lon_indices = []

    lat_bounds = find_index(lats, 'lat')
    lat_indices = [x for x in range(lat_bounds[0], lat_bounds[1]+1)]
    lon_bounds = find_index(lons, 'lon')
    lon_indices = [x for x in range(lon_bounds[0], lon_bounds[1]+1)]

    ascats = []
    iasc = [0, 1]
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    for i in iasc:
        for j in lat_indices:
            for k in lon_indices:
                cut_missing = dataset.variables['nodata'][i][j][k]
                cut_mingmt = dataset.variables['mingmt'][i][j][k]
                cut_land = dataset.variables['land'][i][j][k]
                cut_wspd = dataset.variables['windspd'][i][j][k]
                cut_wdir = dataset.variables['winddir'][i][j][k]
                cut_rain = dataset.variables['scatflag'][i][j][k]

                if (cut_missing
                    or cut_land
                    or cut_wspd == missing_val
                    or cut_wdir == missing_val):
                    pass
                else:
                    ascat = {}
                    ascat['iasc'] = i
                    ascat['lat'] = dataset.variables['latitude'][j]
                    ascat['lon'] = dataset.variables['longitude'][k]
                    ascat['wspd'] = cut_wspd
                    ascat['wdir'] = cut_wdir
                    ascat['time'] = cut_mingmt
                    ascat['rain'] = cut_rain
                    ascat['month'] = month
                    ascat['day'] = day
                    ascats.append(ascat)
    
    return ascats

def main():
    lats = (30, 50)
    lons = (220, 242)
    missing_val = -999.0
    dataset1 = read_ascat('../data/satel/ascat/ascat_20070301_v02.1.gz',
                         missing_val)
    show_dimensions(dataset1)
    show_variables(dataset1)
    show_validrange(dataset1)
    print()
    # (2, 720, 1440)
    print(dataset1.variables['scatflag'].shape)
    ascats = cut_map(dataset1, lats, lons, 3, 1, missing_val)
    # 3620
    print(len(ascats))
    # 0.0
    print(ascats[10]['rain'])

if __name__ == '__main__':
    main()
