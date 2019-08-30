#!/usr/bin/python
#encoding:utf-8
from read_netcdf import ReadNetcdf
#from read_grib import ReadGrib
#from read_bytemaps import ReadBytemap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math

from netCDF4 import Dataset
import read_qscat as rq

lons = (99, 132)
lats = (0, 45)
missing = -32767

def cut_latlon(dataset):
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
    return cut_latitude, cut_longitude, lat_indices, lon_indices

def island(point, ref, D):
    lat_range = (point[0]-D, point[0]+D)
    lon_range = (point[1]-D, point[1]+D)
    lat, lon, lat_ids, lon_ids = cut_latlon(ref)

    for i in lat_ids:
        if ref.variables['latitude'][i] >= lat_range[0] and ref.variables['latitude'][i] <= lat_range[1]:
            for j in lon_ids:
                if ref.variables['longitude'][j] >= lon_range[0] and ref.variables['longitude'][j] <= lon_range[1]:
                    if ref.variables['land'][0][i][j]:
                        return True
    return False

def land_mask(dataset, ref):

    lat_list, lon_list, latid_list, lonid_list = cut_latlon(dataset)
    land = np.zeros((len(lat_list), len(lon_list)))
    lat_index = 0
    for i in lat_list:
        lon_index = 0
        for j in lon_list:
            land[lat_index][lon_index] = island((i, j), ref, 0.75/2)
            lon_index = lon_index + 1
        lat_index = lat_index + 1

    return land


def cut_map(dataset, ref, month):
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

    if month == '02':
        days = 29
    elif month == '04' or month == '06' or month == '09' or month == '11':
        days = 30
    else:
        days = 31

    land = land_mask(dataset, ref)
    # print land.shape
    # print land
    eras = {}
    lat_index = 0
    for i in lat_indices:
        lon_index = 0
        for j in lon_indices:
            if land[lat_index][lon_index]:
                lon_index = lon_index + 1
                continue
            for k in range(days * 8):
                cut_uwnd = dataset.variables['u10'][k][i][j]
                cut_vwnd = dataset.variables['v10'][k][i][j]
                if cut_uwnd == missing or cut_vwnd == missing:
                    continue
                else:
                    cut_wspd = math.sqrt(cut_uwnd**2 + cut_vwnd**2)
                    cut_wdir = math.atan(cut_uwnd / cut_vwnd) * 180 / math.pi
                    if cut_wdir < 0:
                        cut_wdir = cut_wdir + 360
                    if k < (days * 4):
                        hour = (k % 4) * 6
                        day = k / 4 + 1
                    else:
                        hour = ((k-124) % 4) * 6 + 3
                        day = (k-124) / 4 + 1
                    era = {}
                    era['lat'] = dataset.variables['latitude'][i]
                    era['lon'] = dataset.variables['longitude'][j]
                    era['wspd'] = cut_wspd
                    era['wdir'] = cut_wdir
                    era['hour'] = hour
                    era['day'] = day
                    if day not in eras:
                        eras[day] = [era]
                    else:
                        eras[day].append(era)
            lon_index = lon_index + 1
        lat_index = lat_index + 1
    return eras

def read_era(filename):
    dataset = Dataset(filename)
    return dataset

if __name__ == '__main__':
    ref_name = 'datasets/qscat/20081201.gz'
    ref = rq.read_qscat(ref_name)
    filename = 'datasets/era/200812.nc'
    dataset = read_era(filename)
    print 'Dataset read successfully.'
    print 'Dataset format: ', dataset.file_format
    print
    eras = cut_map(dataset, ref)
    print len(eras)


    # print eras
    # for var in dataset.variables:
    #     print var, dataset.variables[var]
