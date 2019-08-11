from quikscat_daily_v4 import QuikScatDaily
import numpy as np
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def read_qscat(filename):
    dataset = QuikScatDaily(filename, missing=missing)
    if not dataset.variables:
        sys.exit('file not found')
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
        if lats[0] <= i <= lats[1]:
            cut_latitude.append(i)
            lat_indices.append(index)
        index = index + 1
    index = 0
    for i in dataset.variables['longitude']:
        if lons[0] <= i <= lons[1]:
            cut_longitude.append(i)
            lon_indices.append(index)
        index = index + 1

    satellites = []
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
                satellite1 = {}
                satellite1['lat'] = dataset.variables['latitude'][i]
                satellite1['lon'] = dataset.variables['longitude'][j]
                satellite1['wspd'] = cut_wspd1
                satellite1['wdir'] = cut_wdir1
                satellite1['time'] = cut_mingmt1
                satellite1['month'] = month
                satellite1['day'] = day
                satellite1['rain'] = cut_rain1
                satellites.append(satellite1)

            if cut_missing2 or cut_land2 or cut_wspd2 == missing or cut_wdir2 == missing:
                pass
            else:
                satellite2 = {}
                satellite2['lat'] = dataset.variables['latitude'][i]
                satellite2['lon'] = dataset.variables['longitude'][j]
                satellite2['wspd'] = cut_wspd2
                satellite2['wdir'] = cut_wdir2
                satellite2['time'] = cut_mingmt2
                satellite2['month'] = month
                satellite2['day'] = day
                satellite2['rain'] = cut_rain2
                satellites.append(satellite2)
    return satellites

def narrow_map(dataset):
    cut_latitude = []
    cut_longitude = []
    lat_indices = []
    lon_indices = []

    index = 0
    for i in dataset.variables['latitude']:
        if lats[0] <= i <= lats[1]:
            cut_latitude.append(i)
            lat_indices.append(index)
        index = index + 1
    index = 0
    for i in dataset.variables['longitude']:
        if lons[0] <= i <= lons[1]:
            cut_longitude.append(i)
            lon_indices.append(index)
        index = index + 1

    wspd1 = np.zeros((len(lat_indices), len(lon_indices)))
    wdir1 = np.zeros((len(lat_indices), len(lon_indices)))
    rain1 = np.zeros((len(lat_indices), len(lon_indices)))
    time1 = np.zeros((len(lat_indices), len(lon_indices)))
    wspd2 = np.zeros((len(lat_indices), len(lon_indices)))
    wdir2 = np.zeros((len(lat_indices), len(lon_indices)))
    rain2 = np.zeros((len(lat_indices), len(lon_indices)))
    time2 = np.zeros((len(lat_indices), len(lon_indices)))

    lat_index = 0
    for i in lat_indices:
        lon_index = 0
        for j in lon_indices:
            wspd1[lat_index][lon_index] = dataset.variables['windspd'][0][i][j]
            wspd2[lat_index][lon_index] = dataset.variables['windspd'][1][i][j]
            wdir1[lat_index][lon_index] = dataset.variables['winddir'][0][i][j]
            wdir2[lat_index][lon_index] = dataset.variables['winddir'][1][i][j]
            rain1[lat_index][lon_index] = dataset.variables['scatflag'][0][i][j]
            rain2[lat_index][lon_index] = dataset.variables['scatflag'][1][i][j]
            time1[lat_index][lon_index] = dataset.variables['mingmt'][0][i][j]
            time2[lat_index][lon_index] = dataset.variables['mingmt'][1][i][j]
            lon_index += 1
        lat_index += 1
    map = {}
    map['wspd'] = [wspd1, wspd2]
    map['wdir'] = [wdir1, wdir2]
    map['rain'] = [rain1, rain2]
    map['time'] = [time1, time2]
    return map


if __name__ == '__main__':
    dataset1 = read_qscat('/Users/zhangdongxiang/PycharmProjects/data4all/qscat/2008/20080101.gz')
    # qscats1 = cut_map(dataset1, 1)
    # print qscats1[1]['hour']
    qscat = narrow_map(dataset1)
    show_dimensions(dataset1)
    show_variables(dataset1)
    show_validrange(dataset1)
    print(qscat['wspd'][0])
    lats = (30, 50)
    lons = (220, 242)

    fig1 = plt.figure(1)
    m = Basemap(projection='mill', llcrnrlon=lons[0], urcrnrlon=lons[1], llcrnrlat=lats[0],
                urcrnrlat=lats[1], resolution='l')
    aximg = m.imshow(qscat['wspd'][0], vmin=0, vmax=25)
    m.drawcoastlines(linewidth=0.25)
    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='#ddaa66', lake_color='#ddaa66')
    parallels = np.arange(lats[0], lats[1], 6.)
    meridians = np.arange(lons[0], lons[1], 7.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6.5, linewidth=0.8)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6.5, linewidth=0.8)
    cbar = fig1.colorbar(aximg, orientation='vertical')
    cbar.set_label('m/s')
    # plt.title('CCMP')
    fig1.suptitle('(a) 10-m Surface Wind Speed (m/s) of CCMP: March 2008')
    plt.show()
    # fig1.savefig('figures/ccmp20083.pdf')
    # print len(qscats1)
    # for i in range(len(qscats1)):
    #     print qscats1[i]['rain']
    # dataset2 = read_qscat('datasets/qscat/20080109.gz')

    # qscats = cut_map(dataset, 1)


