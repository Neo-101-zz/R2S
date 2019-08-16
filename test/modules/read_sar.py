# !/usr/bin/env python

import math
import re
from datetime import date
from datetime import timedelta
import xlrd
import itertools
import csv

import numpy as np
from netCDF4 import Dataset

import netcdf_util

def open_excel(file= 'file.xls'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception as e:
        print(str(e))

def convert_10(wspd, height):
    if wspd <= 7:
        z0 = 0.0023
    else:
        z0 = 0.022
    kz = math.log(10/z0) / math.log(height/z0)
    con_wspd = wspd * kz
    return con_wspd

def excel_table_bygname(file= u'/Users/zhangdongxiang/PycharmProjects/data4all/SAR/data/Buoy/Buoy_data.xls',colnameindex=0,by_name=u'R1_Matchups'):
     data = open_excel(file)
     table = data.sheet_by_name(by_name) #获得表格
     nrows = table.nrows  # 拿到总共行数
     colnames = table.row_values(colnameindex)
     # print(colnames)
     colnames_dict = {k: v for k, v in itertools.zip_longest(colnames, range(len(colnames)))}
     match_info_list = []
     for rownum in range(2, nrows):
         match_info = {}
         match_info['SAR'] = table.cell(rownum, colnames_dict['SAR_Name']).value
         match_info['lat'] = table.cell(rownum, colnames_dict['lat']).value
         match_info['lon'] = table.cell(rownum, colnames_dict['lon']).value
         height = table.cell(rownum, colnames_dict['Ht_B']).value
         wspd = table.cell(rownum, colnames_dict['Ws_B']).value
         match_info['wspd'] = convert_10(wspd, height)
         match_info['buoy'] = table.cell(rownum, colnames_dict['B_id']).value
         match_info_list.append(match_info)
     return match_info_list

base_dir = '/Users/zhangdongxiang/PycharmProjects/data4all/SAR/data/SAR Wind/Radarsat_1_CbandHH/'
"""find the ndbc point"""
# def main():
#     info_list = excel_table_bygname()
#     content = ['SAR', 'buoy', 'min_i', 'min_j', 'wspd', 'ndbc_wspd', 'diff']
#     csvFile = open('/Users/zhangdongxiang/PycharmProjects/data4all/SAR/match2.csv', 'a', newline='')
#     writer = csv.writer(csvFile)
#     writer.writerow(content)
#     csvFile.close()
#     for info in info_list:
#         filename = info['SAR']
#         dataset1 = netcdf_util.ReadNetcdf(base_dir + filename)
#         shape = dataset1.dataset.variables['wind_speed'].shape
#         lat = info['lat']
#         lon = info['lon'] + 360.0
#         ndbc_wspd = info['wspd']
#         buoy = info['buoy']
#         min = 99
#         min_i = 0
#         min_j = 0
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 if (abs(dataset1.dataset.variables['lat'][i][j] - lat)**2 + abs(dataset1.dataset.variables['lon'][i][j] - lon)**2) < min:
#                     min = abs(dataset1.dataset.variables['lat'][i][j] - lat)**2 + abs(dataset1.dataset.variables['lon'][i][j] - lon)**2
#                     # print('bingo!  min: ', min)
#                     min_i = i
#                     min_j = j
#         wspd = dataset1.dataset.variables['wind_speed'][min_i][min_j]
#         # wdir = dataset1.dataset.variables['wind_dir'][min_i][min_j]
#         wspd_diff = wspd - ndbc_wspd
#         # wdir_diff = wdir - ndbc_wdir
#         content = [filename, buoy, min_i, min_j, wspd, ndbc_wspd, wspd_diff]
#         csvFile = open('/Users/zhangdongxiang/PycharmProjects/data4all/SAR/match2.csv', 'a', newline='')
#         writer = csv.writer(csvFile)
#         writer.writerow(content)
#         csvFile.close()
#         # print('min_i: ', min_i)
#         # print('min_j: ', min_j)
#         # print('diff in wspd: ', wspd_diff)
#     # print('diff in wdir: ', wdir_diff)

def main():
    info_list = excel_table_bygname()
    for info in info_list:
        filename = info['SAR']
        dataset1 = netcdf_util.ReadNetcdf(base_dir + filename)
        shape = dataset1.dataset.variables['wind_speed'].shape
        lat = info['lat']
        lon = info['lon'] + 360.0
        ndbc_wspd = info['wspd']
        buoy = info['buoy']
        min = 99
        min_i = 0
        min_j = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (abs(dataset1.dataset.variables['lat'][i][j] - lat)**2 + abs(dataset1.dataset.variables['lon'][i][j] - lon)**2) < min:
                    min = abs(dataset1.dataset.variables['lat'][i][j] - lat)**2 + abs(dataset1.dataset.variables['lon'][i][j] - lon)**2
                    # print('bingo!  min: ', min)
                    min_i = i
                    min_j = j
        wspd = dataset1.dataset.variables['wind_speed'][min_i][min_j]
        # wdir = dataset1.dataset.variables['wind_dir'][min_i][min_j]
        wspd_diff = wspd - ndbc_wspd
        # wdir_diff = wdir - ndbc_wdir
        content = [filename, buoy, min_i, min_j, wspd, ndbc_wspd, wspd_diff]
        csvFile = open('/Users/zhangdongxiang/PycharmProjects/data4all/SAR/match2.csv', 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(content)
        csvFile.close()

if __name__ == '__main__':
    main()
