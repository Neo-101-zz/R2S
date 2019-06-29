#!/usr/bin/python
#encoding:utf-8
import os
import os.path
import gzip
from itertools import islice
import pickle
import math
import pandas as pd
import csv

year_list = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',
             '2012', '2013', '2014', '2015', '2016', '2017']
base_dir = '/Users/zhangdongxiang/PycharmProjects/data4all/ndbc/'
station = {}
exception = ['41036information.txt', 'lscm4information.txt']


lat = (24, 52)
lon = (230, 240)


def read_information(filename):
    with open(filename, 'r') as sf:
        line_list = sf.readlines()
        # line of latitude and longitude
        latlon_line = line_list[3]
        is_north = (latlon_line[7:8] == 'N')
        is_east = (latlon_line[16:17] == 'E')
        if is_north:
            station['lat'] = float(latlon_line[0:6])
        else:
            station['lat'] = -float(latlon_line[0:6])
        if is_east:
            station['lon'] = float(latlon_line[9:15])
        else:
            station['lon'] = 360. - float(latlon_line[9:15])

        # line of measurement height
        site_line = line_list[5]
        if site_line[16:] == 'sea level\n':
            site_height = 0
        else:
            strs = site_line[16:].split(' ')
            site_height = float(strs[0])    # site elevation height
        ane_line = line_list[7]
        strs = ane_line[19:].split(' ')
        wind_height = site_height + float(strs[0])    # anemometer height + site height
        station['height'] = wind_height

def read_gz_file(path):
    if os.path.exists(path):
        with gzip.open(path, 'r') as pf:
            if path == base_dir+'useful_ndbc_data/2008/42002c2008.txt.gz':
                for line in islice(pf, 5624, None):  # skip first 5624 lines
                    yield line
            for line in islice(pf, 2, None):   # skip first 2 lines
                yield line
    else:
        print('the path is not exist!'.format(path))

def is_in(lat, lon):
    if lat[0] <= station['lat'] <= lat[1] and lon[0] <= station['lon'] <= lon[1]:
        return True
    return False

def convert_10(wspd, height):
    if wspd <= 7:
        z0 = 0.0023
    else:
        z0 = 0.022
    kz = math.log(10/z0) / math.log(height/z0)
    con_wspd = wspd * kz
    return con_wspd


if __name__ == '__main__':
    """write all ndbc infomation into csv"""
    # for year in year_list:
    #     print('in: '+ year)
    #     files = os.listdir('./information/'+year)
    #     csvFile = open('./information/' + year + '/info.csv', 'w', newline='')
    #     writer = csv.writer(csvFile)
    #     content = ['id', 'lat', 'lon', 'height']
    #     writer.writerow(content)
    #     csvFile.close()
    #     for file in files:
    #         station['id'] = file[0:5]
    #         if not os.path.isdir(file) and file[0] != '.' and file != 'info.csv':
    #             print('read_information: '+file)
    #             if file in exception:
    #                 continue
    #             read_information('./information/'+year+'/'+file)
    #             if not is_in(lat, lon):
    #                 continue
    #             csvFile = open('./information/'+year+'/info.csv', 'a', newline='')
    #             writer = csv.writer(csvFile)
    #             content = [str(station['id']), str(station['lat']), str(station['lon']), str(station['height'])]
    #             writer.writerow(content)
    #             csvFile.close()
    """read wind data"""
    # for year in year_list:
    #     cur_dir = base_dir+'pickle/'+year
    #     if not os.path.exists(cur_dir):
    #         os.mkdir(cur_dir)
    #     file = pd.read_csv(base_dir+'useful_ndbc_information/'+year+'.csv')
    #     index = 0
    #     for id in file['id']:
    #         station['id'] = id
    #         filename = station['id']+'c'+year+'.txt.gz'
    #         station['height'] = file['height'][index]
    #         station['lat'] = file['lat'][index]
    #         station['lon'] = file['lon'][index]
    #         ndbc = []
    #         print(base_dir+'useful_ndbc_data/'+filename)
    #         line_gen = read_gz_file(base_dir+'useful_ndbc_data/'+year+'/'+filename)
    #         if getattr(line_gen, '__iter__', None):
    #             for line in line_gen:
    #                 line = bytes.decode(line)
    #                 y = line[0:4]      # year
    #                 if y != year:
    #                     continue
    #                 data_point = {}
    #                 data_point['height'] = float(station['height'])
    #                 data_point['lat'] = float(station['lat'])
    #                 data_point['lon'] = float(station['lon'])
    #                 data_point['year'] = y
    #                 data_point['month'] = int(line[5:7])
    #                 data_point['day'] = int(line[8:10])
    #                 data_point['time'] = int(line[11:13]) * 60 + int(line[14:16])
    #                 if year in ['1999', '2000', '2001', '2002', '2003', '2004']:
    #                     data_point['wdir'] = float(line[20:23])
    #                     data_point['wspd'] = convert_10(float(line[26:30]), data_point['height'])
    #                 else:
    #                     data_point['wdir'] = float(line[16:20])
    #                     data_point['wspd'] = convert_10(float(line[21:25]), data_point['height'])
    #                 ndbc.append(data_point)
    #         pickle_file = open(cur_dir + '/'+ id + '_'+ year + '.pkl', 'wb')
    #         pickle.dump(ndbc, pickle_file)
    #         pickle_file.close()
    #         print(id + ' of '+ year +' is okay.')
    #         index += 1
    """end read wind data"""
    pickle_ndbc = open(base_dir+'pickle/1999/46005_1999.pkl', 'rb')
    ndbc = pickle.load(pickle_ndbc)
    print(ndbc)

