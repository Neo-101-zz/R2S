from windsat_daily_v7 import WindSatDaily
import numpy as np
import sys

def read_windsat(filename):
    dataset = WindSatDaily(filename, missing=missing)
    if not dataset.variables: sys.exit('file not found')
    return dataset

lats = (30, 50)
lons = (220, 242)
iasc = 0
wspdname = 'windspd'
wdirname = 'winddir'
missing = -999.

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

def cut_map(dataset, month, day):

    cut_latitude = []
    cut_longitude = []
    lat_indices = []
    lon_indices = []

    index = 0
    for i in dataset.variables['latitude']:
        if i >= lats[0] and i <= lats[1]:
            cut_latitude.append(i)
            lat_indices.append(index)
        index = index + 1
    index = 0
    for i in dataset.variables['longitude']:
        if i >= lons[0] and i <= lons[1]:
            cut_longitude.append(i)
            lon_indices.append(index)
        index = index + 1
    # print lat_indices
    # print lon_indices

    windsats = []
    num = 0
    for i in lat_indices:
        for j in lon_indices:
            num = num + 1
            cut_missing1 = dataset.variables['nodata'][0][i][j]
            cut_mingmt1 = dataset.variables['mingmt'][0][i][j]
            cut_land1 = dataset.variables['land'][0][i][j]
            cut_rain1 = dataset.variables['rain'][0][i][j]
            cut_wspd1_lf = dataset.variables['w-lf'][0][i][j]
            cut_wspd1_mf = dataset.variables['w-mf'][0][i][j]
            cut_wspd1_aw = dataset.variables['w-aw'][0][i][j]
            cut_wdir1 = dataset.variables['wdir'][0][i][j]
            cut_missing2 = dataset.variables['nodata'][1][i][j]
            cut_mingmt2 = dataset.variables['mingmt'][1][i][j]
            cut_land2 = dataset.variables['land'][1][i][j]
            cut_rain2 = dataset.variables['rain'][1][i][j]
            cut_wspd2_lf = dataset.variables['w-lf'][1][i][j]
            cut_wspd2_mf = dataset.variables['w-mf'][1][i][j]
            cut_wspd2_aw = dataset.variables['w-aw'][1][i][j]
            cut_wdir2 = dataset.variables['wdir'][1][i][j]

            
            if cut_missing1 or cut_land1 or cut_wspd1_lf == missing or cut_wspd1_mf == missing or cut_wspd1_aw == missing or cut_wdir1 == missing:
                pass
            else:
                windsat1 = {}
                windsat1['rain'] = cut_rain1
                windsat1['lat'] = dataset.variables['latitude'][i]
                windsat1['lon'] = dataset.variables['longitude'][j]
                windsat1['w-lf'] = cut_wspd1_lf
                windsat1['w-mf'] = cut_wspd1_mf
                windsat1['w-aw'] = cut_wspd1_aw
                windsat1['wdir'] = cut_wdir1
                windsat1['time'] = cut_mingmt1
                windsat1['month'] = month
                windsat1['day'] = day
                windsats.append(windsat1)
            if cut_missing2 or cut_land2 or cut_wspd2_lf == missing or cut_wspd2_mf == missing or cut_wspd2_aw == missing or cut_wdir2 == missing:
                pass
            else:
                windsat2 = {}
                windsat2['rain'] = cut_rain2
                windsat2['lat'] = dataset.variables['latitude'][i]
                windsat2['lon'] = dataset.variables['longitude'][j]
                windsat2['w-lf'] = cut_wspd2_lf
                windsat2['w-mf'] = cut_wspd2_mf
                windsat2['w-aw'] = cut_wspd2_aw
                windsat2['wdir'] = cut_wdir2
                windsat2['time'] = cut_mingmt2
                windsat2['month'] = month
                windsat2['day'] = day
                windsats.append(windsat2)
    # print num
    return windsats

if __name__ == '__main__':
    dataset1 = read_windsat('/Users/zhangdongxiang/PycharmProjects/data4all/windsat/2008/20080101.gz')
    show_dimensions(dataset1)
    show_variables(dataset1)
    show_validrange(dataset1)
    windsats = cut_map(dataset1, 1)
    # print len(windsats)
    # print windsats[0]['rain']
    # print windsats[0]['w-lf']
    # print windsats[0]['w-mf']
    # print windsats[0]['w-aw']
    # print windsats[0]['wdir']
    # print windsats[0]['hour']
    # print windsats[0]['lat']
    # print windsats[0]['lon']
    # print len(windsats)
    # windsat1 = cut_map(dataset1,1)
    # dataset2 = read_windsat('datasets/windsat/20080109.gz')

    # windsats = cut_map(dataset, 1)
