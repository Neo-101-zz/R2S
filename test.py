#!/usr/bin/python
#encoding:utf-8

"""some easy plotting"""
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# import csv
# from itertools import islice
# import os
#
#
#
#
# lat = (24, 52)
# lon = (220, 240)
#
# id_list = []
# lat_list = []
# lon_list = []
#
# with open('/Users/zhangdongxiang/PycharmProjects/data4all/ndbc/useful_ndbc_information/1999.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in islice(reader, 1, None):
#         id = line[0]
#         lat = float(line[1])
#         lon = float(line[2])
#         id_list.append(id)
#         lat_list.append(lat)
#         lon_list.append(lon)
# csvfile.close()
#
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.stock_img()
#
# ny_lon, ny_lat = 100, 10
# delhi_lon, delhi_lat = 77.23, 28.61
#
# plt.scatter(lon_list, lat_list, color='blue', marker='.')
#
#
#
#
#
# plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
#          color='gray', linestyle='--',
#          transform=ccrs.PlateCarree(),
#          )
#
# plt.text(ny_lon - 3, ny_lat - 12, 'New York',
#          horizontalalignment='right',
#          transform=ccrs.Geodetic())
#
# plt.text(delhi_lon + 3, delhi_lat - 12, 'Delhi',
#          horizontalalignment='left',
#          transform=ccrs.Geodetic())
#
# plt.show()
""""finished"""


# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
#
# fig = plt.figure(figsize=(12, 12))
# img_extent = (-77, -59, 9, 26)
#
# ax = plt.axes(projection=ccrs.PlateCarree())
#
# # image data coming from server, code not shown
# ax.stock_img(name='ne_shaded')
# ax.set_xmargin(1)
# ax.set_ymargin(0.10)
#
# # mark a known place to help us geo-locate ourselves
# ax.plot(-117.1625, 32.715, 'bo', markersize=7)
# ax.text(-117, 33, 'San Diego')
#
# # ax.coastlines()
# # ax.gridlines()
#
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from mpl_toolkits.basemap import Basemap
# lats = (0, 45)
# lons = (99, 132)
#
# m = Basemap(projection='mill', lat_ts=10, llcrnrlon=lons[0], urcrnrlon=lons[1], llcrnrlat=lats[0],
#                     urcrnrlat=lats[1], resolution='c')
# m.drawcoastlines(linewidth=0.25)
# m.drawcountries(linewidth=0.25)
# m.fillcontinents(color='#ddaa66', lake_color='#ddaa66')
# parallels = np.arange(lats[0], lats[1], 6.)
# meridians = np.arange(lons[0], lons[1], 7.)
# m.drawparallels(parallels, labels=[1,0,0,0], fontsize=6.5, linewidth=0.8)
# m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=6.5, linewidth=0.8)
#
# proj = ccrs.Mercator()
# fig = plt.figure(figsize=(12, 12))
# extents = proj.transform_points(ccrs.Geodetic(),
#                                 np.array([-77, -59]),
#                                 np.array([9, 26]))
#
# img_extents = (extents[0][0], extents[1][0], extents[0][6], extents[1][7] )
#
# ax = plt.axes(projection=proj)
# # image data coming from server, code not shown
# ax.imshow(img, origin='upper', extent=img_extents,transform=proj)
#
# ax.set_xmargin(0.05)
# ax.set_ymargin(0.10)
#
# # mark a known place to help us geo-locate ourselves
# ax.plot(-117.1625, 32.715, 'bo', markersize=7, transform=ccrs.Geodetic())
# ax.text(-117, 33, 'San Diego', transform=ccrs.Geodetic())
#
#
#
# ax.coastlines()
# ax.gridlines()
#
# plt.show()


"""information for all NDBCs"""
# year_list = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',
#              '2012', '2013', '2014', '2015', '2016', '2017']
#
# files = os.listdir('/Users/zhangdongxiang/Desktop/师姐工作总结/NDBC浮标坐标和高度信息')
# csvFile = open('/Users/zhangdongxiang/Desktop/师姐工作总结/NDBC浮标坐标和高度信息/all_info.csv', 'w', newline='')
# writer = csv.writer(csvFile)
# content = ['id', 'lat', 'lon', 'height', 'years']
# writer.writerow(content)
# csvFile.close()
# info = {}
# id_list = []
# for year in year_list:
#     with open('/Users/zhangdongxiang/Desktop/师姐工作总结/NDBC浮标坐标和高度信息/'+year+'.csv', 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         for line in islice(reader, 1, None):
#             id = line[0]
#             if id in id_list:
#                 info[id]['years'] += '+'+year
#             else:
#                 id_list.append(id)
#                 info[id] = {'id': id, 'lat': line[1], 'lon': line[2], 'height': line[3], 'years': year}
# for read in info:
#     content = [info[read]['id'], info[read]['lat'], info[read]['lon'], info[read]['height'], info[read]['years']]
#     csvFile = open('/Users/zhangdongxiang/Desktop/师姐工作总结/NDBC浮标坐标和高度信息/all_info.csv', 'a', newline='')
#     writer = csv.writer(csvFile)
#     writer.writerow(content)
# csvFile.close()



"""change files' name"""
#
# import os
# # years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
# # for year in years:
# dir = '/Users/zhangdongxiang/Downloads/as'
# files = os.listdir(dir)
# for file in files:
#     new_name = file[6:14] + '.gz'
#     print(new_name)
#
#     os.rename(dir+ '/' + file, dir +'/' + new_name)


"""select useful files"""
# import os
# import shutil
# import pandas as pd
#
# years = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
# for year in years:
#     dir = '/Users/zhangdongxiang/PycharmProjects/data4all/ndbc/'
#     os.mkdir(dir+'useful_ndbc_data/'+year)
#     file = pd.read_csv(dir+'useful_ndbc_information/'+year+'.csv')
#     for id in file['id']:
#         oldfile = dir+'data/'+year+'/'+id+'c'+year+'.txt.gz'
#         # oldfile = '/Users/zhangdongxiang/PycharmProjects/data4all/ndbc/data/1999/46050c1999.txt.gz'
#         print(oldfile)
#         newfile = dir+'useful_ndbc_data/'+year+'/'+id+'c'+year+'.txt.gz'
#         shutil.copyfile(oldfile, newfile)


"""download ERA-Interim"""
#
# from ecmwfapi import ECMWFDataServer
#
# base_dir = '/Users/zhangdongxiang/PycharmProjects/data4all/'
# year_list = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
# server = ECMWFDataServer()
# for year in year_list:
#     server.retrieve({
#         "class": "ei",
#         "dataset": "interim",
#         "date": year+"-01-01/to/"+year+"-12-31",
#         "expver": "1",
#         "grid": "0.125/0.125",
#         "levtype": "sfc",
#         "param": "165.128/166.128",
#         "area": "50/220/30/242",
#         "step": "0",
#         "stream": "oper",
#         "domain": "G",
#         "time": "00:00:00/06:00:00/12:00:00/18:00:00",
#         "type": "an",
#         'format': "netcdf",
#         "target": base_dir+"era-interim/"+year+".nc",
#     })
"""mkdir for years"""
# qscat_year_list = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
# ascat_year_list = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
# windsat_year_list = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
#
# import os
#
# for year in qscat_year_list:
#     if not os.path.exists('/Users/zhangdongxiang/PycharmProjects/data4all/match/figure/qscat/'+year):
#         os.mkdir('/Users/zhangdongxiang/PycharmProjects/data4all/match/figure/qscat/'+year)
#
# for year in ascat_year_list:
#     if not os.path.exists('/Users/zhangdongxiang/PycharmProjects/data4all/match/figure/ascat/'+year):
#         os.mkdir('/Users/zhangdongxiang/PycharmProjects/data4all/match/figure/ascat/'+year)
#


"""Ridge Regression test"""

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def ridgeRegres(xMat, yMat, lam=0.2):
#     xTx = xMat.T * xMat
#     denom = xTx + np.eye(np.shape(xMat)[1]) * lam
#     if np.linalg.det(denom) == 0.0:
#         print("This matrix is singular, cannot do inverse")
#         return
#     ws = denom.I * (xMat.T * yMat)
#     return ws
#
#
# def ridgeTest(xArr, yArr):
#     xMat = np.mat(xArr)
#     yMat = np.mat(yArr).T
#     yMean = np.mean(yMat)  # 数据标准化
#     # print(yMean)
#     yMat = yMat - yMean
#     # print(xMat)
#     # regularize X's
#     xMeans = np.mean(xMat, 0)
#     xVar = np.var(xMat, 0)
#     xMat = (xMat - xMeans) / xVar  # （特征-均值）/方差
#     numTestPts = 30
#     wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
#     for i in range(numTestPts):  # 测试不同的lambda取值，获得系数
#         ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
#         wMat[i, :] = ws.T
#     return wMat
#
#
# # import data
# ex0 = np.loadtxt('abalone.txt', delimiter='\t')
# xArr = ex0[:, 0:-1]
# yArr = ex0[:, -1]
# # print(xArr,yArr)
# ridgeWeights = ridgeTest(xArr, yArr)
# # print(ridgeWeights)
# plt.plot(ridgeWeights)
# plt.show()

"""Some plot"""
# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

# names = ['[0,5)', '[5,10)', '[10,15]', '[15,+'+'$\infty$)']
# x = range(len(names))
qscat = [68.954,34.542,21.879,27.162]
ascat = [63.395,30.726,20.815,17.745]
windsat = [44.803,23.309,13.721,14.925]

# names = [0.92,1.34,17.94,18.99,20.05,20.33,24.31,26.79,28.13,28.36,28.67,29.32,32.21,33.64,34.35,35.29,39.1,136.44,485.49,531.95,540.25,1078.94] #ascat
# names = [0.92,1.34,17.8,17.94,18.99,20.05,20.33,24.31,26.79,28.13,28.36,28.67,29.32,32.21,33.64,34.35,35.29,39.1,136.44,485.49,531.95,540.25,1078.94] #qscat
names = [136.44,485.49,531.95,540.25,1078.94] #windsat
x = range(len(names))
# mbe = [1.678,0.874,-0.839,0.695,-0.521,-0.175,-0.555,-0.618,-0.086,0.254,0.317,0.52,-0.129,-0.24,0.36,0.053,-0.469,0.276,-0.421,-0.435,-0.429,-0.275]
# mae = [2.366,1.687,1.971,1.655,1.31,1.382,0.969,1.156,0.876,0.905,0.784,1.168,0.889,1.031,1.00,0.806,0.935,0.893,0.883,0.849,0.778,0.803]
# rmse = [3.395,2.331,2.402,2.275,1.65,1.815,1.27,1.453,1.167,1.302,1.013,1.588,1.262,1.36,1.15,1.091,1.211,1.223,1.145,1.115,1.03,1.067]
# cc = [0.778,0.879,0.864,0.864,0.937,0.92,0.959,0.936,0.946,0.944,0.961,0.905,0.945,0.944,0.945,0.953,0.968,0.942,0.963,0.966,0.968,0.965]

# mae = [41.689,30.152,23.882,26.646,20.682,25.394,14.636,14.667,20.825,24.603,19.647,22.259,16.767,18.007,22.123,23.065,18.263,12.206,11.351,13.601,13.133,11.227]
# rmse = [64.618,52.823,42.499,47.421,39.862,47.15,29.759,31.602,41.627,45.365,35.627,41.221,33.281,35.197,36.914,40.223,37.331,23.778,22.115,21.865,20.735,20.615]

# mae_wspd = [2.133,1.896,1.449,2.462,1.892,1.532,1.77,1.218,1.128,1.153,1.178,1.018,1.052,1.043,1.248,1.312,1.219,0.976,0.84,1.024,1.013,0.848,1.08]
# rmse_wspd = [3.047,2.498,1.982,3.197,2.545,2.011,2.327,1.624,1.451,1.544,1.676,1.37,1.475,1.518,1.774,1.739,1.7,1.317,1.31,1.467,1.337,1.188,1.407]
# cc_wspd = [0.826,0.859,0.864,0.796,0.831,0.899,0.88,0.926,0.928,0.905,0.906,0.934,0.872,0.91,0.896,0.844,0.901,0.958,0.937,0.928,0.954,0.952,0.959]
#
# mae_wdir = [43.813,34.796,29.755,33.583,33.634,27.57,30.677,20.754,20.792,25.353,27.77,23.877,28.258,22.708,26.401,29.629,27.432,24.116,14.241,16.903,13.651,16.005,12.246]
# rmse_wdir = [65.04,56.303,48.905,46.617,53.928,47.167,50.546,38.336,39.342,43.871,46.747,40.431,47.533,40.052,45.478,48.75,45.846,42.124,27.023,31.442,25.977,26.542,24.499]

rmse_wspd = [1.313,1.358,1.231,1.143,1.35]
mae_wspd = [0.998,1.026,0.958,0.916,1.056]
cc_wspd = [0.906,0.927,0.951,0.953,0.947]

mae_wdir = [15.552,15.666,18.831,16.189,15.966]
rmse_wdir = [24.809,25.128,31.454,24.015,24.903]

names = [1,2,3,4]

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
fig = plt.figure()

plt.plot(x, mae_wspd, marker='*', ms=10,label=u'MBE(m/s)')
# plt.plot(x, mae_wdir, marker='*', ms=10,label=u'MAE('+r'$^\circ$'+')')
plt.plot(x, rmse_wspd, marker='o', mec='r', mfc='w',label=u'RMSE(m/s)')
# plt.plot(x, rmse_wdir, marker='o', mec='r', mfc='w',label=u'RMSE('+r'$^\circ$'+')')
plt.plot(x, cc_wspd, marker='x', ms=10,label=u'CC')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45, fontsize=5)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Distance to Coast(km)") #X轴标签
plt.ylabel('MAE and RMSE(m/s)') #Y轴标签
# plt.ylabel('MAE and RMSE('+r'$^\circ$'+')') #Y轴标签
# plt.title("A simple plot") #标题

fig.savefig(u"/Users/zhangdongxiang/Documents/2018硕士毕设/图片/第三章/windsat_离岸适用性_wspd.pdf")