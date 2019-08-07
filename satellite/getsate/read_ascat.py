from ascat_daily import ASCATDaily
import numpy as np
import sys

def read_ascat(filename):
    dataset = ASCATDaily(filename, missing=missing)
    if not dataset.variables: sys.exit('file not found')
    return dataset

lats = (30, 50)
lons = (220, 242)

iasc = 0
wspdname = 'windspd'
wdirname = 'winddir'
missing = -999.0

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

    ascats = []
    num = 0
    for i in lat_indices:
        for j in lon_indices:
            num = num + 1
            cut_missing1 = dataset.variables['nodata'][0][i][j]
            cut_mingmt1 = dataset.variables['mingmt'][0][i][j]
            cut_land1 = dataset.variables['land'][0][i][j]
            cut_wspd1 = dataset.variables['windspd'][0][i][j]
            cut_wdir1 = dataset.variables['winddir'][0][i][j]
            cut_rain1 = dataset.variables['scatflag'][0][i][j]
            cut_missing2 = dataset.variables['nodata'][1][i][j]
            cut_mingmt2 = dataset.variables['mingmt'][1][i][j]
            cut_land2 = dataset.variables['land'][1][i][j]
            cut_wspd2 = dataset.variables['windspd'][1][i][j]
            cut_wdir2 = dataset.variables['winddir'][1][i][j]
            cut_rain2 = dataset.variables['scatflag'][1][i][j]

            if cut_missing1 or cut_land1 or cut_wspd1 == missing or cut_wdir1 == missing:
                pass
            else:
                ascat1 = {}
                ascat1['lat'] = dataset.variables['latitude'][i]
                ascat1['lon'] = dataset.variables['longitude'][j]
                ascat1['wspd'] = cut_wspd1
                ascat1['wdir'] = cut_wdir1
                ascat1['time'] = cut_mingmt1
                ascat1['rain'] = cut_rain1
                ascat1['month'] = month
                ascat1['day'] = day
                ascats.append(ascat1)
            if cut_missing2 or cut_land2 or cut_wspd2 == missing or cut_wdir2 == missing:
                pass
            else:
                ascat2 = {}
                ascat2['lat'] = dataset.variables['latitude'][i]
                ascat2['lon'] = dataset.variables['longitude'][j]
                ascat2['wspd'] = cut_wspd2
                ascat2['wdir'] = cut_wdir2
                ascat2['time'] = cut_mingmt2
                ascat2['rain'] = cut_rain2
                ascat2['month'] = month
                ascat2['day'] = day
                ascats.append(ascat2)
    # print num
    return ascats

if __name__ == '__main__':
    dataset1 = read_ascat('datasets/ascat/20080101.gz')
    show_dimensions(dataset1)
    show_variables(dataset1)
    show_validrange(dataset1)
    print(dataset1.variables['scatflag'].shape)
    ascats1 = cut_map(dataset1, 1)
    print(ascats1[10]['rain'])
    # for i in range(len(ascats1)):
    #     print ascats1[i]['wspd']
    # print ascats1[0]['wdir']
    # print ascats1[0]['hour']
    # print ascats1[0]['lat']
    # print ascats1[0]['lon']
    
    # print len(ascats)
    # ascat1 = cut_map(dataset1,1)
    # dataset2 = read_ascat('datasets/ascat/20080109.gz')

    # ascats = cut_map(dataset, 1)
