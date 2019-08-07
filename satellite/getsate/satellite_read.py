import read_qscat as rq
import read_ascat as ra
import read_windsat as rw
import os

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

qscat_year_list = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
ascat_year_list = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
windsat_year_list = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
base_dir = '/Users/zhangdongxiang/PycharmProjects/data4all/'

# def batch_read_qscat(month):
#     # miss = ('datasets/qscat/20081126.gz',
#     #         'datasets/qscat/20081127.gz')
#     # if month == '02':
#     #     days = 29
#     # elif month == '04' or month == '06' or month == '09' or month == '11':
#     #     days = 30
#     # else:
#     #     days = 31
#     qscats = {}
#     for i in range(days):
#         day = i + 1
#         if i <= 8:
#             filename = 'datasets/qscat/2008' + month + '0' + str(i + 1) + '.gz'
#         else:
#             filename = 'datasets/qscat/2008' + month + str(i + 1) + '.gz'
#         if filename in miss:
#             qscats[day] = []
#             continue
#         dataset = rq.read_qscat(filename)
#         qscats[day] = rq.cut_map(dataset, day)
#     return qscats
#
# def batch_read_ascat(month):
#     miss = ('datasets/ascat/20080117.gz',
#             'datasets/ascat/20080320.gz')
#     if month == '02':
#         days = 29
#     elif month == '04' or month == '06' or month == '09' or month == '11':
#         days = 30
#     else:
#         days = 31
#     ascats = {}
#     for i in range(days):
#         day = i + 1
#         if i <= 8:
#             filename = 'datasets/ascat/2008' + month + '0' + str(i + 1) + '.gz'
#         else:
#             filename = 'datasets/ascat/2008' + month + str(i + 1) + '.gz'
#         if filename in miss:
#             ascats[day] = []
#             continue
#         dataset = ra.read_ascat(filename)
#         ascats[day] = ra.cut_map(dataset, day)
#     return ascats
#
# def batch_read_windsat(month):
#     miss = ('datasets/windsat/20080229.gz',
#             'datasets/windsat/20080301.gz',
#             'datasets/windsat/20080302.gz',
#             'datasets/windsat/20080329.gz',
#             'datasets/windsat/20080330.gz',
#             'datasets/windsat/20080402.gz',
#             'datasets/windsat/20080403.gz',
#             'datasets/windsat/20080518.gz',
#             'datasets/windsat/20080610.gz',
#             'datasets/windsat/20080611.gz',
#             'datasets/windsat/20080612.gz',
#             'datasets/windsat/20080613.gz',
#             'datasets/windsat/20080614.gz',
#             'datasets/windsat/20080615.gz',
#             'datasets/windsat/20080616.gz',
#             'datasets/windsat/20080617.gz',
#             'datasets/windsat/20080618.gz',
#             'datasets/windsat/20080619.gz',
#             'datasets/windsat/20080620.gz',
#             'datasets/windsat/20080621.gz',
#             'datasets/windsat/20080622.gz',
#             'datasets/windsat/20080623.gz',
#             'datasets/windsat/20080624.gz',
#             'datasets/windsat/20080625.gz',
#             'datasets/windsat/20080626.gz',
#             'datasets/windsat/20080627.gz',
#             'datasets/windsat/20080628.gz',
#             'datasets/windsat/20080629.gz',
#             'datasets/windsat/20080630.gz',
#             'datasets/windsat/20080727.gz',
#             'datasets/windsat/20081127.gz',
#             'datasets/windsat/20081216.gz')
#     if month == '02':
#         days = 29
#     elif month == '04' or month == '06' or month == '09' or month == '11':
#         days = 30
#     else:
#         days = 31
#     windsats = {}
#     for i in range(days):
#         day = i + 1
#         if i <= 8:
#             filename = 'datasets/windsat/2008' + month + '0' + str(i + 1) + '.gz'
#         else:
#             filename = 'datasets/windsat/2008' + month + str(i + 1) + '.gz'
#         if filename in miss:
#             windsats[day] = []
#             continue
#         dataset = rw.read_windsat(filename)
#         windsats[day] = rw.cut_map(dataset, day)
#     return windsats

if __name__ == '__main__':
    for year in windsat_year_list:
        dir = base_dir + 'windsat/' + year
        files = os.listdir(dir)
        files.sort()
        cur_dir = base_dir + 'windsat/pickle/' + year
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

        last_month = 0
        windsats_for_month = []
        for file in files:
            print(file)
            if file[0] not in ['1', '2']:
                continue
            month = int(file[4:6])
            day = int(file[6:8])
            dataset = rw.read_windsat(dir+'/'+file)
            if month == last_month:
                windsats_for_month += rw.cut_map(dataset, month, day)
            else:
                if last_month == 0:
                    windsats_for_month = []
                    windsats_for_month += rw.cut_map(dataset, month, day)
                    last_month = month
                else:
                    pickle_file = open(cur_dir + '/windsat_' + str(last_month) + '.pkl', 'wb')
                    pickle.dump(windsats_for_month, pickle_file)
                    pickle_file.close()
                    windsats_for_month = []
                    windsats_for_month += rw.cut_map(dataset, month, day)
                    last_month = month
        pickle_file = open(cur_dir + '/windsat_' + str(last_month) + '.pkl', 'wb')
        pickle.dump(windsats_for_month, pickle_file)
        pickle_file.close()




