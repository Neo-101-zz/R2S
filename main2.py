#!/usr/bin/python
#encoding:utf-8
import read_qscat as rq
import read_ascat as ra
import read_windsat as rw
import read_ncp as rn

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from scipy.stats import gaussian_kde

from sklearn import linear_model


def match1(dataset_x, dataset_y, space_d, time_d):
    wspd_list_x = []
    wspd_list_y = []
    wdir_list_x = []
    wdir_list_y = []
    if len(dataset_x) and len(dataset_y):
        for x in dataset_x:
            for y in dataset_y:
                if abs(x['lat']-y['lat']) < space_d and abs(x['lon']-y['lon']) < space_d and abs(x['hour'] - y['hour']) < time_d:
                    wspd_list_x.append(x['wspd'])
                    wspd_list_y.append(y['wspd'])
                    wdir_list_x.append(x['wdir'])
                    wdir_list_y.append(y['wdir'])
                    # match_point1 = (x['wspd'], y['wspd'])
                    # match_point2 = (x['wdir'], y['wdir'])
                    # match_list.append((match_point1, match_point2))
    return wspd_list_x, wspd_list_y, wdir_list_x, wdir_list_y

def match2(dataset_x, dataset_y, space_d, time_d):
    wspd_list_x = []
    wspdlf_list_y = []
    wspdmf_list_y = []
    wspdaw_list_y = []
    wdir_list_x = []
    wdir_list_y = []
    if len(dataset_x) and len(dataset_y):
        for x in dataset_x:
            for y in dataset_y:
                if abs(x['lat']-y['lat']) < space_d and abs(x['lon']-y['lon']) < space_d and abs(x['hour'] - y['hour']) < time_d:
                    wspd_list_x.append(x['wspd'])
                    wspdlf_list_y.append(y['w-lf'])
                    wspdmf_list_y.append(y['w-mf'])
                    wspdaw_list_y.append(y['w-aw'])
                    wdir_list_x.append(x['wdir'])
                    wdir_list_y.append(y['wdir'])
                    # match_point1 = (x['wspd'], y['wspd'])
                    # match_point2 = (x['wdir'], y['wdir'])
                    # match_list.append((match_point1, match_point2))
    return wspd_list_x, wspdlf_list_y, wspdmf_list_y, wspdaw_list_y, wdir_list_x, wdir_list_y

def match3(dataset_x, dataset_y, space_d, time_d):
    wspd_list_x = []
    wspd_list_y = []
    wdir_list_x = []
    wdir_list_y = []
    rain_list = []
    if len(dataset_x) and len(dataset_y):
        for x in dataset_x:
            for y in dataset_y:
                if abs(x['lat']-y['lat']) < space_d and abs(x['lon']-y['lon']) < space_d and abs(x['hour'] - y['hour']) < time_d:
                    wspd_list_x.append(x['wspd'])
                    wspd_list_y.append(y['wspd'])
                    wdir_list_x.append(x['wdir'])
                    wdir_list_y.append(y['wdir'])
                    rain_list.append(y['rain'])
                    # match_point1 = (x['wspd'], y['wspd'])
                    # match_point2 = (x['wdir'], y['wdir'])
                    # match_list.append((match_point1, match_point2))
    return wspd_list_x, wspd_list_y, wdir_list_x, wdir_list_y, rain_list

def match_new(ships, qscats, ascats, windsats, eras):
    match_list = []
    if len(ships) and len(qscats) and len(ascats) and len(windsats) and len(eras):
        print len(ships)
        print len(qscats)
        print len(ascats)
        print len(windsats)
        print len(eras)
        for ship in ships:
            lat = ship['lat']
            lon = ship['lon']
            hour = ship['hour']
            qscat_flag = 0
            qscat_wait = {}
            for qscat in qscats:
                if abs(qscat['lat'] - lat) < 0.125 and abs(qscat['lat'] - lon) < 0.125 and abs(qscat['hour'] - hour) < 1:
                    qscat_flag = 1
                    qscat_wait = qscat
                    break
            ascat_flag = 0
            ascat_wait = {}
            for ascat in ascats:
                if abs(ascat['lat'] - lat) < 0.125 and abs(ascat['lat'] - lon) < 0.125 and abs(ascat['hour'] - hour) < 1:
                    ascat_flag = 1
                    ascat_wait = ascat
                    break
            windsat_flag = 0
            windsat_wait = {}
            for windsat in windsats:
                if abs(windsat['lat'] - lat) < 0.125 and abs(windsat['lat'] - lon) < 0.125 and abs(windsat['hour'] - hour) < 1:
                    windsat_flag = 1
                    windsat_wait = windsat
                    break
            era_flag = 0
            era_wait = {}
            for era in eras:
                if abs(era['lat'] - lat) < 0.375 and abs(era['lat'] - lon) < 0.375 and abs(era['hour'] - hour) < 1:
                    era_flag = 1
                    era_wait = era
                    break
            if (qscat_flag + ascat_flag + windsat_flag + era_flag) >= 2:
                match = {}
                match['ship'] = ship
                match['qscat'] = qscat_wait
                match['ascat'] = ascat_wait
                match['windsat'] = windsat_wait
                match['era'] = era_wait
                match_list.append(match)
    return match_list

def read_situ(filename):
    with open(filename, 'r') as f:
        situs = {}
        for line in f:
            lat = int(line[25:31])
            lon = int(line[31:38])
            wdir = line[60:64]
            wspd = line[64:68]
            sdir = line[125:129]
            sspd = line[129:132]
            if lat >= 0 and lat <= 4500 and lon >= 9900 and lon <= 13200 and wdir != ' ' * 4 and wspd != ' ' * 4 and sdir != ' ' * 4 and sspd != ' ' * 3:
                id = line[0:5]
                day = int(line[19:22])
                hour = int(line[22:25])
                wdir = int(wdir)
                wspd = int(wspd)
                sdir = int(sdir)
                sspd = int(sspd)
                # print 'sdir: ', sdir
                # print 'sspd: ', sspd
                vwnd = wspd/math.sqrt((math.tan(wdir))**2 + 1)
                uwnd = vwnd * math.tan(wdir)
                vsnd = sspd/math.sqrt((math.tan(sdir))**2 + 1)
                usnd = vsnd * math.tan(sdir)
                vwnd = vwnd - vsnd
                uwnd = uwnd - usnd
                wspd = math.sqrt(uwnd ** 2 + vwnd ** 2)
                if vwnd == 0 and uwnd < 0:
                    wdir = 270
                elif vwnd == 0 and uwnd > 0:
                    wdir = 90
                elif vwnd == 0 and uwnd == 0:
                    wdir =0
                else:
                    wdir = math.atan(uwnd / vwnd) * 180 / math.pi
                    if wdir < 0:
                        wdir = wdir + 360
                situ = {'id': id, 'lat': lat / 100., 'lon': lon / 100., 'day': day, 'hour': hour, 'wdir': wdir,
                        'wspd': wspd}
                if day not in situs:
                    situs[day] = [situ]
                else:
                    situs[day].append(situ)
    return situs

def batch_read_qscat(month):
    miss = ('datasets/qscat/20081126.gz',
            'datasets/qscat/20081127.gz')
    if month == '02':
        days = 29
    elif month == '04' or month == '06' or month == '09' or month == '11':
        days = 30
    else:
        days = 31
    qscats = {}
    for i in range(days):
        day = i + 1
        if i <= 8:
            filename = 'datasets/qscat/2008' + month + '0' + str(i + 1) + '.gz'
        else:
            filename = 'datasets/qscat/2008' + month + str(i + 1) + '.gz'
        if filename in miss:
            qscats[day] = []
            continue
        dataset = rq.read_qscat(filename)
        qscats[day] = rq.cut_map(dataset, day)
    return qscats

def batch_read_ascat(month):
    miss = ('datasets/ascat/20080117.gz',
            'datasets/ascat/20080320.gz')
    if month == '02':
        days = 29
    elif month == '04' or month == '06' or month == '09' or month == '11':
        days = 30
    else:
        days = 31
    ascats = {}
    for i in range(days):
        day = i + 1
        if i <= 8:
            filename = 'datasets/ascat/2008' + month + '0' + str(i + 1) + '.gz'
        else:
            filename = 'datasets/ascat/2008' + month + str(i + 1) + '.gz'
        if filename in miss:
            ascats[day] = []
            continue
        dataset = ra.read_ascat(filename)
        ascats[day] = ra.cut_map(dataset, day)
    return ascats

def batch_read_windsat(month):
    miss = ('datasets/windsat/20080229.gz',
            'datasets/windsat/20080301.gz',
            'datasets/windsat/20080302.gz',
            'datasets/windsat/20080329.gz',
            'datasets/windsat/20080330.gz',
            'datasets/windsat/20080402.gz',
            'datasets/windsat/20080403.gz',
            'datasets/windsat/20080518.gz',
            'datasets/windsat/20080610.gz',
            'datasets/windsat/20080611.gz',
            'datasets/windsat/20080612.gz',
            'datasets/windsat/20080613.gz',
            'datasets/windsat/20080614.gz',
            'datasets/windsat/20080615.gz',
            'datasets/windsat/20080616.gz',
            'datasets/windsat/20080617.gz',
            'datasets/windsat/20080618.gz',
            'datasets/windsat/20080619.gz',
            'datasets/windsat/20080620.gz',
            'datasets/windsat/20080621.gz',
            'datasets/windsat/20080622.gz',
            'datasets/windsat/20080623.gz',
            'datasets/windsat/20080624.gz',
            'datasets/windsat/20080625.gz',
            'datasets/windsat/20080626.gz',
            'datasets/windsat/20080627.gz',
            'datasets/windsat/20080628.gz',
            'datasets/windsat/20080629.gz',
            'datasets/windsat/20080630.gz',
            'datasets/windsat/20080727.gz',
            'datasets/windsat/20081127.gz',
            'datasets/windsat/20081216.gz')
    if month == '02':
        days = 29
    elif month == '04' or month == '06' or month == '09' or month == '11':
        days = 30
    else:
        days = 31
    windsats = {}
    for i in range(days):
        day = i + 1
        if i <= 8:
            filename = 'datasets/windsat/2008' + month + '0' + str(i + 1) + '.gz'
        else:
            filename = 'datasets/windsat/2008' + month + str(i + 1) + '.gz'
        if filename in miss:
            windsats[day] = []
            continue
        dataset = rw.read_windsat(filename)
        windsats[day] = rw.cut_map(dataset, day)
    return windsats

def cal_mean_bias(x_list, y_list):
    bias_list = list(map(lambda x: x[1] - x[0], zip(x_list, y_list)))
    # print("%s\n%s\n%s" % (x_list, y_list, bias_list))
    bias_mean = sum(bias_list) / len(bias_list)
    return bias_mean

def cal_rmse(x_list, y_list):
    rmse_list = list(map(lambda x: (x[1] - x[0])**2, zip(x_list, y_list)))
    # print("%s\n%s\n%s" % (x_list, y_list, rmse_list))
    rmse = math.sqrt(sum(rmse_list) / len(rmse_list))
    return rmse

def cal_mean(x):
    return sum(x) / len(x)

def cal_std(x):
    mean = cal_mean(x)
    std_list = []
    for i in x:
        std_list.append((i - mean)**2)
    if len(std_list) == 1:
        return 0.0000000001
    std = math.sqrt(sum(std_list) / (len(std_list) - 1))
    return std

def cal_co(x_list, y_list):
    x_mean = cal_mean(x_list)
    y_mean = cal_mean(y_list)
    x_std = cal_std(x_list)
    y_std = cal_std(y_list)
    co_list = []
    length = len(x_list)
    for i in range(length):
        co_list.append((x_list[i] - x_mean) * (y_list[i] - y_mean) / (x_std * y_std))
    co = sum(co_list) / len(co_list)
    return co

def plot_scatter_wspd1(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 34, 5))
    plt.yticks(range(0, 34, 5))
    plt.text(0.5, 27, content)
    plt.xlabel('Ship wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig('Fig/'+name+'.eps')
    fig.savefig('Fig/'+name+'.pdf')

def plot_scatter_wspd2(x1, x2, x3, y1, y2, y3, content, name, sate):
    # regr1 = linear_model.LinearRegression()
    # regr2 = linear_model.LinearRegression()
    # regr3 = linear_model.LinearRegression()
    # regr1.fit(np.array(x).reshape(-1, 1), y1)
    # # a, b = regr1.coef_, regr1.intercept_
    # regr2.fit(np.array(x).reshape(-1, 1), y2)
    # regr3.fit(np.array(x).reshape(-1, 1), y3)
    fig = plt.figure()
    xy1 = np.vstack([x1, y1])
    density1 = gaussian_kde(xy1)(xy1)
    xy2 = np.vstack([x2, y2])
    density2 = gaussian_kde(xy2)(xy2)
    xy3 = np.vstack([x3, y3])
    density3 = gaussian_kde(xy3)(xy3)
    plt.scatter(x1, y1, c=density1, marker='_', label="low frequency", linewidths=3)
    plt.scatter(x2, y2, c=density2, marker='|', label="medium frequency", linewidths=2)
    plt.scatter(x3, y3, c=density3, marker='x', label="all weather", linewidths=1)
    plt.plot(x3, x3, color='black', linewidth=0.5, alpha = 0.2)
    # plt.plot(x, regr1.predict(np.array(x).reshape(-1, 1)), color='red', linestyle='--', linewidth=3)
    # plt.plot(x, regr2.predict(np.array(x).reshape(-1, 1)), color='green', linestyle='--', linewidth=2)
    # plt.plot(x, regr3.predict(np.array(x).reshape(-1, 1)), color='blue', linestyle='--', linewidth=1)
    plt.legend(loc='upper right', fontsize=9)
    plt.xticks(range(0, 41, 5))
    plt.yticks(range(0, 41, 5))
    plt.xlabel('Ship wind speed (m/s)')
    plt.ylabel(sate+' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    plt.text(0, 29.5, content, size=9, weight="light")
    # plt.title(name)
    # plt.show()
    fig.savefig('Fig/'+name+'.eps')
    fig.savefig('Fig/'+name+'.pdf')

def plot_scatter_wspd3(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 41, 5))
    plt.yticks(range(0, 41, 5))
    plt.text(0, 34, content)
    plt.xlabel('Ship wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig('Fig/'+name+'.eps')
    fig.savefig('Fig/'+name+'.pdf')

def plot_scatter_wspd4(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 34, 5))
    plt.yticks(range(0, 34, 5))
    plt.text(0, 26, content)
    plt.xlabel('Ship wind speed (m/s)')
    plt.ylabel(sate + ' wind speed without rain (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig('Fig/'+name+'.eps')
    fig.savefig('Fig/'+name+'.pdf')

def plot_scatter_wdir(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_
    x_change = [(i+180)%360 for i in x]
    y_change = [(i+180)%360 for i in y]
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    fig = plt.figure()
    plt.scatter(x_change, y_change, c=density, marker=".")
    plt.plot(range(361), range(361), color='black', linewidth=0.5, alpha = 0.2)
    plt.plot(range(180,361), [(i-180) for i in range(180,361)], color='black', linewidth=0.5, linestyle='--', alpha = 0.2)
    plt.plot(range(181), [(i+180) for i in range(181)],color='black', linewidth=0.5, linestyle='--', alpha = 0.2)
    plt.xticks(range(0, 361, 60), ['180', '240', '300', '0', '60', '120', '180'])
    plt.yticks(range(0, 361, 60), ['180', '240', '300', '0', '60', '120', '180'])
    plt.xlabel('Vessel wind direction ('+ r'$^\circ$'+ ')')
    plt.ylabel(sate + ' wind direction ('+ r'$^\circ$'+ ')')
    plt.text(0, 300, content)
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    fig.savefig('Fig/' + name + '.eps')
    fig.savefig('Fig/' + name + '.pdf')

def plot_scatter_wdir4(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    x_change = [(i+180)%360 for i in x]
    y_change = [(i+180)%360 for i in y]
    fig = plt.figure()
    plt.scatter(x_change, y_change, c=density, marker=".")
    plt.plot(range(361), range(361), color='black', linewidth=0.5, alpha = 0.2)
    plt.plot(range(180,361), [(i-180) for i in range(180,361)], color='black', linewidth=0.5, linestyle='--', alpha = 0.2)
    plt.plot(range(181), [(i+180) for i in range(181)],color='black', linewidth=0.5, linestyle='--', alpha = 0.2)
    plt.xticks(range(0, 361, 60), ['180', '240', '300', '0', '60', '120', '180'])
    plt.yticks(range(0, 361, 60), ['180', '240', '300', '0', '60', '120', '180'])
    plt.xlabel('Vessel wind direction ('+ r'$^\circ$'+ ')')
    plt.ylabel(sate + ' wind direction without rain ('+ r'$^\circ$'+ ')')
    plt.text(0, 300, content)
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    fig.savefig('Fig/' + name + '.eps')
    fig.savefig('Fig/' + name + '.pdf')

def filt(xlist, ylist, limit):
    a = []
    b = []
    for i in range(len(xlist)):
        if abs(xlist[i] - ylist[i]) < limit:
            a.append(xlist[i])
            b.append(ylist[i])
    return a, b

def cut(xlist, ylist):
    a = []
    b = []
    for i in range(len(xlist)):
        if xlist[i] <= 360:
            a.append(xlist[i])
            b.append(ylist[i])
    for i in range(len(a)):
        if a[i] == 360:
            a[i] = 0
        # if b[i] - a[i] >= 180:
        #     b[i] = b[i] - 180
        # elif a[i] - b[i] >= 180:
        #     b[i] = b[i] + 180
        if b[i] == 360:
            b[i] = 0
    return a, b

def change(xlist, ylist):
    a = xlist
    b = ylist
    for i in range(len(a)):
        if a[i] == 360:
            a[i] = 0
        if b[i] - a[i] >= 180:
            b[i] = b[i] - 180
        elif a[i] - b[i] >= 180:
            b[i] = b[i] + 180
        if b[i] == 360:
            b[i] = 0
    return a, b

def divide(xwspd, ywspd, xwdir, ywdir):
    low_xwspd = []
    low_ywspd = []
    mid_xwspd = []
    mid_ywspd = []
    high_xwspd = []
    high_ywspd = []
    low_xwdir = []
    low_ywdir = []
    mid_xwdir = []
    mid_ywdir = []
    high_xwdir = []
    high_ywdir = []
    for i in range(len(ywspd)):
        if ywspd[i] < 5:
            low_xwspd.append(xwspd[i])
            low_ywspd.append(ywspd[i])
            low_xwdir.append(xwdir[i])
            low_ywdir.append(ywdir[i])
        elif ywspd[i] > 15:
            high_xwspd.append(xwspd[i])
            high_ywspd.append(ywspd[i])
            high_xwdir.append(xwdir[i])
            high_ywdir.append(ywdir[i])
        else:
            mid_xwspd.append(xwspd[i])
            mid_ywspd.append(ywspd[i])
            mid_xwdir.append(xwdir[i])
            mid_ywdir.append(ywdir[i])
    return low_xwspd, low_ywspd, mid_xwspd, mid_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid_xwdir, mid_ywdir, high_xwdir, high_ywdir

def divide2(xwspd, ywspd, xwdir, ywdir):
    low_xwspd = []
    low_ywspd = []
    mid1_xwspd = []
    mid1_ywspd = []
    mid2_xwspd = []
    mid2_ywspd = []
    high_xwspd = []
    high_ywspd = []
    low_xwdir = []
    low_ywdir = []
    mid1_xwdir = []
    mid1_ywdir = []
    mid2_xwdir = []
    mid2_ywdir = []
    high_xwdir = []
    high_ywdir = []
    for i in range(len(ywspd)):
        if ywspd[i] < 5:
            low_xwspd.append(xwspd[i])
            low_ywspd.append(ywspd[i])
            low_xwdir.append(xwdir[i])
            low_ywdir.append(ywdir[i])
        elif ywspd[i] > 15:
            high_xwspd.append(xwspd[i])
            high_ywspd.append(ywspd[i])
            high_xwdir.append(xwdir[i])
            high_ywdir.append(ywdir[i])
        elif ywspd[i] > 10 and ywspd[i] <= 15:
            mid2_xwspd.append(xwspd[i])
            mid2_ywspd.append(ywspd[i])
            mid2_xwdir.append(xwdir[i])
            mid2_ywdir.append(ywdir[i])
        else:
            mid1_xwspd.append(xwspd[i])
            mid1_ywspd.append(ywspd[i])
            mid1_xwdir.append(xwdir[i])
            mid1_ywdir.append(ywdir[i])

    return low_xwspd, low_ywspd, mid1_xwspd, mid1_ywspd, mid2_xwspd, mid2_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid1_xwdir, mid1_ywdir, mid2_xwdir, mid2_ywdir, high_xwdir, high_ywdir

def rain_cut(dict):
    urain = {'wspd_x': [], 'wspd_y': [], 'wdir_x': [], 'wdir_y': []}
    length = len(dict['rain'])
    for i in range(length):
        if dict['rain'][i] == 0:
            urain['wspd_x'].append(dict['wspd_x'][i])
            urain['wspd_y'].append(dict['wspd_y'][i])
            urain['wdir_x'].append(dict['wdir_x'][i])
            urain['wdir_y'].append(dict['wdir_y'][i])
    return urain


if __name__ == '__main__':
    """Start to read!"""
    month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    situs_list = {}
    qscats_list = {}
    ascats_list = {}
    windsats_list = {}
    eras_list = {}
    ref_name = 'datasets/qscat/20081201.gz'
    ref = rq.read_qscat(ref_name)

    for month in month_list:
        if month == '02':
            days = 29
        elif month == '04' or month == '06' or month == '09' or month == '11':
            days = 30
        else:
            days = 31
        situs = read_situ(filename='datasets/in-situ/OCEN_SHB_GLB_FTM_SHIP-2008' + month + '.TXT')
        qscats = batch_read_qscat(month)
        ascats = batch_read_ascat(month)
        windsats = batch_read_windsat(month)
        print 'Start to read ERA: ' + month
        dataset = rn.read_era(filename='datasets/era/2008' + month + '.nc')
        eras = rn.cut_map(dataset, ref, month)
        situs_list[month] = situs
        eras_list[month] = eras
        situs_list[month] = situs
        qscats_list[month] = qscats
        ascats_list[month] = ascats
        windsats_list[month] = windsats
        print 'Data reading ' + month + ' finished!'
    #
    # wspd_lists_x = []
    # wspd_lists_y = []
    # wdir_lists_x = []
    # wdir_lists_y = []
    # rain_lists = []
    # num = 0
    # for month in month_list:
    #     if month == '02':
    #         days = 29
    #     elif month == '04' or month == '06' or month == '09' or month == '11':
    #         days = 30
    #     else:
    #         days = 31
    #     # qscats = batch_read_qscat(month)
    #     # print
    #     # print 'QSCAT '+month+' read finished! Start to match!!!!'
    #     # print
    #     for i in range(days):
    #         wspd_list_x, wspd_list_y, wdir_list_x, wdir_list_y, rain_list = match3(situs_list[month][i + 1],
    #                                                                     qscats_list[month][i + 1], 0.25/2, 1)
    #         wspd_lists_x = wspd_lists_x + wspd_list_x
    #         wspd_lists_y = wspd_lists_y + wspd_list_y
    #         wdir_lists_x = wdir_lists_x + wdir_list_x
    #         wdir_lists_y = wdir_lists_y + wdir_list_y
    #         rain_lists = rain_lists + rain_list
    #     num = len(wspd_lists_x)
    #     print month + 'QSCAT: Current num: ', num
    # match_dataset = {}
    # match_dataset['wspd_x'] = wspd_lists_x
    # match_dataset['wspd_y'] = wspd_lists_y
    # match_dataset['wdir_x'] = wdir_lists_x
    # match_dataset['wdir_y'] = wdir_lists_y
    # match_dataset['rain'] = rain_lists
    # pickle_file = open('pickle/qscat.pkl', 'wb')
    # pickle.dump(match_dataset, pickle_file)
    # pickle_file.close()
    # print 'QSCAT finished!'
    #
    # wspd_lists_x = []
    # wspd_lists_y = []
    # wdir_lists_x = []
    # wdir_lists_y = []
    # rain_lists = []
    # num = 0
    # for month in month_list:
    #     if month == '02':
    #         days = 29
    #     elif month == '04' or month == '06' or month == '09' or month == '11':
    #         days = 30
    #     else:
    #         days = 31
    #     # qscats = batch_read_qscat(month)
    #     # print
    #     # print 'QSCAT '+month+' read finished! Start to match!!!!'
    #     # print
    #     for i in range(days):
    #         wspd_list_x, wspd_list_y, wdir_list_x, wdir_list_y, rain_list = match3(situs_list[month][i + 1],
    #                                                                     ascats_list[month][i + 1], 0.25/2, 1)
    #         wspd_lists_x = wspd_lists_x + wspd_list_x
    #         wspd_lists_y = wspd_lists_y + wspd_list_y
    #         wdir_lists_x = wdir_lists_x + wdir_list_x
    #         wdir_lists_y = wdir_lists_y + wdir_list_y
    #         rain_lists = rain_lists + rain_list
    #     num = len(wspd_lists_x)
    #     print month + 'ASCAT: Current num: ', num
    # match_dataset = {}
    # match_dataset['wspd_x'] = wspd_lists_x
    # match_dataset['wspd_y'] = wspd_lists_y
    # match_dataset['wdir_x'] = wdir_lists_x
    # match_dataset['wdir_y'] = wdir_lists_y
    # match_dataset['rain'] = rain_lists
    # pickle_file = open('pickle/ascat.pkl', 'wb')
    # pickle.dump(match_dataset, pickle_file)
    # pickle_file.close()
    # print 'ASCAT finished!'
    #
    # wspd_lists_x = []
    # wspdlf_lists_y = []
    # wspdmf_lists_y = []
    # wspdaw_lists_y = []
    # wdir_lists_x = []
    # wdir_lists_y = []
    # num = 0
    # for month in month_list:
    #     if month == '02':
    #         days = 29
    #     elif month == '04' or month == '06' or month == '09' or month == '11':
    #         days = 30
    #     else:
    #         days = 31
    #     for i in range(days):
    #         wspd_list_x, wspdlf_list_y, wspdmf_list_y, wspdaw_list_y, wdir_list_x, wdir_list_y = match2(
    #             situs_list[month][i + 1], windsats_list[month][i + 1], 0.25/2, 1)
    #         wspd_lists_x = wspd_lists_x + wspd_list_x
    #         wspdlf_lists_y = wspdlf_lists_y + wspdlf_list_y
    #         wspdmf_lists_y = wspdmf_lists_y + wspdmf_list_y
    #         wspdaw_lists_y = wspdaw_lists_y + wspdaw_list_y
    #         wdir_lists_x = wdir_lists_x + wdir_list_x
    #         wdir_lists_y = wdir_lists_y + wdir_list_y
    #     num = len(wspd_lists_x)
    #     print month + 'WindSat: Current num: ', num
    # match_dataset = {}
    # match_dataset['wspd_x'] = wspd_lists_x
    # match_dataset['wspdlf_y'] = wspdlf_lists_y
    # match_dataset['wspdmf_y'] = wspdmf_lists_y
    # match_dataset['wspdaw_y'] = wspdaw_lists_y
    # match_dataset['wdir_x'] = wdir_lists_x
    # match_dataset['wdir_y'] = wdir_lists_y
    # pickle_file = open('pickle/windsat.pkl', 'wb')
    # pickle.dump(match_dataset, pickle_file)
    # pickle_file.close()
    # print 'WindSat finished!'
    #
    # wspd_lists_x = []
    # wspd_lists_y = []
    # wdir_lists_x = []
    # wdir_lists_y = []
    # num = 0
    # for month in month_list:
    #     if month == '02':
    #         days = 29
    #     elif month == '04' or month == '06' or month == '09' or month == '11':
    #         days = 30
    #     else:
    #         days = 31
    #     # qscats = batch_read_qscat(month)
    #     # print
    #     # print 'QSCAT '+month+' read finished! Start to match!!!!'
    #     # print
    #     for i in range(days):
    #         wspd_list_x, wspd_list_y, wdir_list_x, wdir_list_y = match1(situs_list[month][i + 1],
    #                                                                     eras_list[month][i + 1], 0.75 / 2, 1)
    #         wspd_lists_x = wspd_lists_x + wspd_list_x
    #         wspd_lists_y = wspd_lists_y + wspd_list_y
    #         wdir_lists_x = wdir_lists_x + wdir_list_x
    #         wdir_lists_y = wdir_lists_y + wdir_list_y
    #     num = len(wspd_lists_x)
    #     print month + 'ERA: Current num: ', num
    # match_dataset = {}
    # match_dataset['wspd_x'] = wspd_lists_x
    # match_dataset['wspd_y'] = wspd_lists_y
    # match_dataset['wdir_x'] = wdir_lists_x
    # match_dataset['wdir_y'] = wdir_lists_y
    # pickle_file = open('pickle/era.pkl', 'wb')
    # pickle.dump(match_dataset, pickle_file)
    # pickle_file.close()
    # print 'ERA finished!'
    """Reading finished!"""
    
    """plot: """
    # """qscat:"""
    # qscat_file = open('pickle/qscat.pkl', 'rb')
    # qscat = pickle.load(qscat_file)
    # wspd_x = qscat['wspd_x']
    # wspd_y = qscat['wspd_y']
    # # wdir_x = qscat['wdir_x']
    # # wdir_y = qscat['wdir_y']
    # number = len(qscat['wspd_y'])
    # wspd_x, wspd_y = filt(wspd_x, wspd_y, limit=20)
    # # wdir_x, wdir_y = cut(wdir_x, wdir_y)
    # wspd_mean_bias = cal_mean_bias(wspd_x, wspd_y)
    # wspd_rmse = cal_rmse(wspd_x, wspd_y)
    # wspd_co = cal_co(wspd_x, wspd_y)
    # # wdir_mean_bias = cal_mean_bias(wdir_x, wdir_y)
    # # wdir_rmse = cal_rmse(wdir_x, wdir_y)
    # # wdir_co = cal_co(wdir_x, wdir_y)
    # # wspd_content = 'Mean Bias (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    # #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    # #                '\nR: ' + str(round(wspd_co, 3)) + \
    # #                '\nNumber of Matches: ' + str(len(wspd_x))
    # # wdir_content = 'Mean Bias ('+r'$^\circ$'+'): ' + str(round(wdir_mean_bias, 3)) + \
    # #                '\nRMSE ('+r'$^\circ$'+'): ' + str(round(wdir_rmse, 3)) + \
    # #                '\nR: ' + str(round(wdir_co, 3)) + \
    # #                '\nNumber of Matches: ' + str(len(wdir_x))
    # wspd_content = 'Mean Bias (m/s): 0.193' + \
    #                '\nRMSE (m/s): 4.164'  + \
    #                '\nNumber of Matches: 1085'
    # # wdir_content = 'Mean Bias (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    # #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    # #                '\nR: ' + str(round(wdir_co, 3)) + \
    # #                '\nNumber of Matches: 1085'
    # plot_scatter_wspd1(wspd_x, wspd_y, content=wspd_content, name='qscat_wspd', sate='QSCAT')
    # # plot_scatter_wdir(wdir_x, wdir_y, content=wdir_content, name='qscat_wdir', sate='QSCAT')

    # """ascat:"""
    # ascat_file = open('pickle/ascat.pkl', 'rb')
    # ascat = pickle.load(ascat_file)
    # wspd_x = ascat['wspd_x']
    # wspd_y = ascat['wspd_y']
    # # wdir_x = ascat['wdir_x']
    # # wdir_y = ascat['wdir_y']
    # wspd_x, wspd_y = filt(wspd_x, wspd_y, limit=20)
    # # wdir_x, wdir_y = cut(wdir_x, wdir_y)
    # wspd_mean_bias = cal_mean_bias(wspd_x, wspd_y)
    # wspd_rmse = cal_rmse(wspd_x, wspd_y)
    # wspd_co = cal_co(wspd_x, wspd_y)
    # # wdir_mean_bias = cal_mean_bias(wdir_x, wdir_y)
    # # wdir_rmse = cal_rmse(wdir_x, wdir_y)
    # # wdir_co = cal_co(wdir_x, wdir_y)
    # # wspd_content = 'Mean Bias (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    # #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    # #                '\nR: ' + str(round(wspd_co, 3)) + \
    # #                '\nNumber of Matches: ' + str(len(wspd_x))
    # # wdir_content = 'Mean Bias ('+r'$^\circ$'+'): ' + str(round(wdir_mean_bias, 3)) + \
    # #                '\nRMSE ('+r'$^\circ$'+'): ' + str(round(wdir_rmse, 3)) + \
    # #                '\nR: ' + str(round(wdir_co, 3)) + \
    # #                '\nNumber of Matches: ' + str(len(wdir_x))
    # wspd_content = 'Mean Bias (m/s): 0.163' + \
    #                '\nRMSE (m/s): 4.148' + \
    #                '\nNumber of Matches: 1490'
    # # wdir_content = 'Mean Bias (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    # #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    # #                '\nR: ' + str(round(wdir_co, 3)) + \
    # #                '\nNumber of Matches: 1490'
    plot_scatter_wspd1(wspd_x, wspd_y, content=wspd_content, name='ascat_wspd', sate='ASCAT')
    plot_scatter_wdir(wdir_x, wdir_y, content=wdir_content, name='ascat_wdir', sate='ASCAT')
    #
    # """windsat:"""
    # windsat_file = open('pickle/windsat.pkl', 'rb')
    # windsat = pickle.load(windsat_file)
    # # wspdlf_x = windsat['wspd_x']
    # # wspdmf_x = windsat['wspd_x']
    # wspdaw_x = windsat['wspd_x']
    # # wspdlf_y = windsat['wspdlf_y']
    # # wspdmf_y = windsat['wspdmf_y']
    # wspdaw_y = windsat['wspdaw_y']
    # # wdir_x = windsat['wdir_x']
    # # wdir_y = windsat['wdir_y']
    # # wspdlf_x, wspdlf_y = filt(wspdlf_x, wspdlf_y, limit=20)
    # # wspdmf_x, wspdmf_y = filt(wspdmf_x, wspdmf_y, limit=20)
    # wspdaw_x, wspdaw_y = filt(wspdaw_x, wspdaw_y, limit=20)
    # # wdir_x, wdir_y = cut(wdir_x, wdir_y)
    # # wspdlf_mean_bias = cal_mean_bias(wspdlf_x, wspdlf_y)
    # # wspdlf_rmse = cal_rmse(wspdlf_x, wspdlf_y)
    # # wspdlf_co = cal_co(wspdlf_x, wspdlf_y)
    # # wspdmf_mean_bias = cal_mean_bias(wspdmf_x, wspdmf_y)
    # # wspdmf_rmse = cal_rmse(wspdmf_x, wspdmf_y)
    # # wspdmf_co = cal_co(wspdmf_x, wspdmf_y)
    # # wspdaw_mean_bias = cal_mean_bias(wspdaw_x, wspdaw_y)
    # # wspdaw_rmse = cal_rmse(wspdaw_x, wspdaw_y)
    # # wspdaw_co = cal_co(wspdaw_x, wspdaw_y)
    # # wdir_mean_bias = cal_mean_bias(wdir_x, wdir_y)
    # # wdir_rmse = cal_rmse(wdir_x, wdir_y)
    # # wdir_co = cal_co(wdir_x, wdir_y)
    # # wspd_content = 'Mean Bias (m/s): \n  lf: ' + str(round(wspdlf_mean_bias, 3)) + \
    # #                '; mf: ' + str(round(wspdmf_mean_bias, 3)) + \
    # #                '; aw: ' + str(round(wspdaw_mean_bias, 3)) + \
    # #                '\nRMSE (m/s): \n  lf: ' + str(round(wspdlf_rmse, 3)) + \
    # #                '; mf: ' + str(round(wspdmf_rmse, 3)) + \
    # #                '; aw: ' + str(round(wspdaw_rmse, 3)) + \
    # #                '\nR: \n  lf: ' + str(round(wspdlf_co, 3)) + \
    # #                '; mf: ' + str(round(wspdmf_co, 3)) + \
    # #                '; aw: ' + str(round(wspdaw_co, 3)) + \
    # #                '\nNumber of Matches: \n  lf: ' + str(len(wspdlf_x)) + \
    # #                '; mf: ' + str(len(wspdmf_x)) + \
    # #                '; aw: ' + str(len(wspdaw_x))
    # # wdir_content = 'Mean Bias ('+r'$^\circ$'+'): ' + str(round(wdir_mean_bias, 3)) + \
    # #                '\nRMSE ('+r'$^\circ$'+'): ' + str(round(wdir_rmse, 3)) + \
    # #                '\nR: ' + str(round(wdir_co, 3)) + \
    # #                '\nNumber of Matches: ' + str(len(wdir_x))
    # # wspd_content = 'Mean Bias (m/s): \n  lf: -0.208' + \
    # #                '; mf: -0.318' + \
    # #                '; aw: -0.168' + \
    # #                '\nRMSE (m/s): \n  lf: 4.358'  + \
    # #                '; mf: 4.352' + \
    # #                '; aw: 4.362' + \
    # #                '\nNumber of Matches: \n  lf: 442' + \
    # #                '; mf: 442' + \
    # #                '; aw: 442'
    # # wdir_content = 'Mean Bias (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    # #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    # #                '\nR: ' + str(round(wdir_co, 3)) + \
    # #                '\nNumber of Matches: 446'
    # wspd_content = 'Mean Bias (m/s): -0.168' + \
    #                '\nRMSE (m/s): 4.362' + \
    #                '\nNumber of Matches: 442'
    # # wdir_content = 'Mean Bias (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    # #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    # #                '\nR: ' + str(round(wdir_co, 3)) + \
    # #                '\nNumber of Matches: 1085'
    # # plot_scatter_wspd1(wspdlf_x, wspdmf_x, wspdaw_x, wspdlf_y, wspdmf_y, wspdaw_y, content=wspd_content, name='windsat_wspd', sate='WindSat')
    # plot_scatter_wspd1(wspdaw_x, wspdaw_y, content=wspd_content, name='windsat_wspd', sate='WindSat')
    # # plot_scatter_wdir(wdir_x, wdir_y, content=wdir_content, name='windsat_wdir', sate='WindSat')

    """ERA"""
    # era_file = open('pickle/era.pkl', 'rb')
    # era = pickle.load(era_file)
    # wspd_x = era['wspd_x']
    # wspd_y = era['wspd_y']
    # wdir_x = era['wdir_x']
    # wdir_y = era['wdir_y']
    # wspd_x, wspd_y = filt(wspd_x, wspd_y, limit=20)
    # wdir_x, wdir_y = cut(wdir_x, wdir_y)
    # wspd_mean_bias = cal_mean_bias(wspd_x, wspd_y)
    # wspd_rmse = cal_rmse(wspd_x, wspd_y)
    # wspd_co = cal_co(wspd_x, wspd_y)
    # wdir_mean_bias = cal_mean_bias(wdir_x, wdir_y)
    # wdir_rmse = cal_rmse(wdir_x, wdir_y)
    # wdir_co = cal_co(wdir_x, wdir_y)
    # wspd_content = 'Mean Bias (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nR: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wspd_x))
    # wdir_content = 'Mean Bias (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wdir_x))
    # wspd_content = 'Mean Bias (m/s): -0.991' + \
    #                '\nRMSE (m/s): 4.606' + \
    #                '\nNumber of Matches: 35212'
    # wdir_content = 'Mean Bias (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: 0.181' + \
    #                '\nNumber of Matches: 35212'
    # plot_scatter_wspd3(wspd_x, wspd_y, content=wspd_content, name='era_wspd', sate='ERA-Interim')
    # plot_scatter_wdir(wdir_x, wdir_y, content=wdir_content, name='era_wdir', sate='ERA-Interim')

    """Calculate divide"""
    # """qscat"""
    # qscat_file = open('pickle/qscat.pkl', 'rb')
    # qscat = pickle.load(qscat_file)
    # wspd_x = qscat['wspd_x']
    # wspd_y = qscat['wspd_y']
    # wdir_x = qscat['wdir_x']
    # wdir_y = qscat['wdir_y']
    # # wspd_x, wspd_y = filt(wspd_x, wspd_y, limit=20)
    # # wdir_x, wdir_y = cut_and_change(wdir_x, wdir_y)
    # # wdir_x, wdir_y = change(wdir_x, wdir_y)
    # # low_xwspd, low_ywspd, mid_xwspd, mid_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid_xwdir, mid_ywdir, high_xwdir, high_ywdir = divide(
    # #     wspd_x, wspd_y, wdir_x, wdir_y)
    # low_xwspd, low_ywspd, mid1_xwspd, mid1_ywspd, mid2_xwspd, mid2_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid1_xwdir, mid1_ywdir, mid2_xwdir, mid2_ywdir, high_xwdir, high_ywdir = divide2(wspd_x, wspd_y, wdir_x, wdir_y)
    # low_xwspd, low_ywspd = filt(low_xwspd, low_ywspd, 20)
    # mid1_xwspd, mid1_ywspd = filt(mid1_xwspd, mid1_ywspd, 20)
    # mid2_xwspd, mid2_ywspd = filt(mid2_xwspd, mid2_ywspd, 20)
    # high_xwspd, high_ywspd = filt(high_xwspd, high_ywspd, 20)
    # # low_xwdir, low_ywdir = cut_and_change(low_xwdir, low_ywdir)
    # # mid_xwdir, mid_ywdir = cut_and_change(mid_xwdir, mid_ywdir)
    # # high_xwdir, high_ywdir = cut_and_change(high_xwdir, high_ywdir)
    # low_wspd_mean_bias = cal_mean_bias(low_xwspd, low_ywspd)
    # low_wspd_rmse = cal_rmse(low_xwspd, low_ywspd)
    # low_wspd_co = cal_co(low_xwspd, low_ywspd)
    # mid1_wspd_mean_bias = cal_mean_bias(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_rmse = cal_rmse(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_co = cal_co(mid1_xwspd, mid1_ywspd)
    # mid2_wspd_mean_bias = cal_mean_bias(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_rmse = cal_rmse(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_co = cal_co(mid2_xwspd, mid2_ywspd)
    # high_wspd_mean_bias = cal_mean_bias(high_xwspd, high_ywspd)
    # high_wspd_rmse = cal_rmse(high_xwspd, high_ywspd)
    # high_wspd_co = cal_co(high_xwspd, high_ywspd)
    # low_wdir_mean_bias = cal_mean_bias(low_xwdir, low_ywdir)
    # low_wdir_rmse = cal_rmse(low_xwdir, low_ywdir)
    # low_wdir_co = cal_co(low_xwdir, low_ywdir)
    # # mid_wdir_mean_bias = cal_mean_bias(mid_xwdir, mid_ywdir)
    # # mid_wdir_rmse = cal_rmse(mid_xwdir, mid_ywdir)
    # # mid_wdir_co = cal_co(mid_xwdir, mid_ywdir)
    # # high_wdir_mean_bias = cal_mean_bias(high_xwdir, high_ywdir)
    # # high_wdir_rmse = cal_rmse(high_xwdir, high_ywdir)
    # # high_wdir_co = cal_co(high_xwdir, high_ywdir)
    # print 'qscat: '
    # print 'low:'
    # print 'low_wspd_mean_bias: ', low_wspd_mean_bias
    # print 'low_wspd_rmse: ', low_wspd_rmse
    # print 'low_wspd_co: ', low_wspd_co
    # # print 'low_wdir_mean_bias', low_wdir_mean_bias
    # # print 'low_wdir_rmse: ', low_wdir_rmse
    # # print 'low_wdir_co: ', low_wdir_co
    # print 'N: ', len(low_ywspd)
    # print 'mid1:'
    # print 'mid1_wspd_mean_bias: ', mid1_wspd_mean_bias
    # print 'mid1_wspd_rmse: ', mid1_wspd_rmse
    # print 'mid1_wspd_co: ', mid1_wspd_co
    # # print 'mid_wdir_mean_bias', mid_wdir_mean_bias
    # # print 'mid_wdir_rmse: ', mid_wdir_rmse
    # # print 'mid_wdir_co: ', mid_wdir_co
    # print 'N: ', len(mid1_ywspd)
    # print 'mid2:'
    # print 'mid2_wspd_mean_bias: ', mid2_wspd_mean_bias
    # print 'mid2_wspd_rmse: ', mid2_wspd_rmse
    # print 'mid2_wspd_co: ', mid2_wspd_co
    # print 'N: ', len(mid2_ywspd)
    # print 'high:'
    # print 'high_wspd_mean_bias: ', high_wspd_mean_bias
    # print 'high_wspd_rmse: ', high_wspd_rmse
    # print 'high_wspd_co: ', high_wspd_co
    # # print 'high_wdir_mean_bias', high_wdir_mean_bias
    # # print 'high_wdir_rmse: ', high_wdir_rmse
    # # print 'high_wdir_co: ', high_wdir_co
    # print 'N: ', len(high_ywspd)
    #
    # """ascat"""
    # ascat_file = open('pickle/ascat.pkl', 'rb')
    # ascat = pickle.load(ascat_file)
    # wspd_x = ascat['wspd_x']
    # wspd_y = ascat['wspd_y']
    # wdir_x = ascat['wdir_x']
    # wdir_y = ascat['wdir_y']
    # # wspd_x, wspd_y = filt(wspd_x, wspd_y, limit=20)
    # # wdir_x, wdir_y = cut_and_change(wdir_x, wdir_y)
    # # wdir_x, wdir_y = change(wdir_x, wdir_y)
    # low_xwspd, low_ywspd, mid1_xwspd, mid1_ywspd, mid2_xwspd, mid2_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid1_xwdir, mid1_ywdir, mid2_xwdir, mid2_ywdir, high_xwdir, high_ywdir = divide2(wspd_x, wspd_y, wdir_x, wdir_y)
    # low_xwspd, low_ywspd = filt(low_xwspd, low_ywspd, 20)
    # mid1_xwspd, mid1_ywspd = filt(mid1_xwspd, mid1_ywspd, 20)
    # mid2_xwspd, mid2_ywspd = filt(mid2_xwspd, mid2_ywspd, 20)
    # high_xwspd, high_ywspd = filt(high_xwspd, high_ywspd, 20)
    # # low_xwdir, low_ywdir = cut_and_change(low_xwdir, low_ywdir)
    # # mid_xwdir, mid_ywdir = cut_and_change(mid_xwdir, mid_ywdir)
    # # high_xwdir, high_ywdir = cut_and_change(high_xwdir, high_ywdir)
    # low_wspd_mean_bias = cal_mean_bias(low_xwspd, low_ywspd)
    # low_wspd_rmse = cal_rmse(low_xwspd, low_ywspd)
    # low_wspd_co = cal_co(low_xwspd, low_ywspd)
    # mid1_wspd_mean_bias = cal_mean_bias(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_rmse = cal_rmse(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_co = cal_co(mid1_xwspd, mid1_ywspd)
    # mid2_wspd_mean_bias = cal_mean_bias(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_rmse = cal_rmse(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_co = cal_co(mid2_xwspd, mid2_ywspd)
    # high_wspd_mean_bias = cal_mean_bias(high_xwspd, high_ywspd)
    # high_wspd_rmse = cal_rmse(high_xwspd, high_ywspd)
    # high_wspd_co = cal_co(high_xwspd, high_ywspd)
    # # low_wdir_mean_bias = cal_mean_bias(low_xwdir, low_ywdir)
    # # low_wdir_rmse = cal_rmse(low_xwdir, low_ywdir)
    # # low_wdir_co = cal_co(low_xwdir, low_ywdir)
    # # mid_wdir_mean_bias = cal_mean_bias(mid_xwdir, mid_ywdir)
    # # mid_wdir_rmse = cal_rmse(mid_xwdir, mid_ywdir)
    # # mid_wdir_co = cal_co(mid_xwdir, mid_ywdir)
    # # high_wdir_mean_bias = cal_mean_bias(high_xwdir, high_ywdir)
    # # high_wdir_rmse = cal_rmse(high_xwdir, high_ywdir)
    # # high_wdir_co = cal_co(high_xwdir, high_ywdir)
    # print 'ascat: '
    # print 'low:'
    # print 'low_wspd_mean_bias: ', low_wspd_mean_bias
    # print 'low_wspd_rmse: ', low_wspd_rmse
    # print 'low_wspd_co: ', low_wspd_co
    # # print 'low_wdir_mean_bias', low_wdir_mean_bias
    # # print 'low_wdir_rmse: ', low_wdir_rmse
    # # print 'low_wdir_co: ', low_wdir_co
    # print 'N: ', len(low_ywspd)
    # print 'mid1:'
    # print 'mid1_wspd_mean_bias: ', mid1_wspd_mean_bias
    # print 'mid1_wspd_rmse: ', mid1_wspd_rmse
    # print 'mid1_wspd_co: ', mid1_wspd_co
    # # print 'mid_wdir_mean_bias', mid_wdir_mean_bias
    # # print 'mid_wdir_rmse: ', mid_wdir_rmse
    # # print 'mid_wdir_co: ', mid_wdir_co
    # print 'N: ', len(mid1_ywspd)
    # print 'mid2:'
    # print 'mid2_wspd_mean_bias: ', mid2_wspd_mean_bias
    # print 'mid2_wspd_rmse: ', mid2_wspd_rmse
    # print 'mid2_wspd_co: ', mid2_wspd_co
    # print 'N: ', len(mid2_ywspd)
    # print 'high:'
    # print 'high_wspd_mean_bias: ', high_wspd_mean_bias
    # print 'high_wspd_rmse: ', high_wspd_rmse
    # print 'high_wspd_co: ', high_wspd_co
    # # print 'high_wdir_mean_bias', high_wdir_mean_bias
    # # print 'high_wdir_rmse: ', high_wdir_rmse
    # # print 'high_wdir_co: ', high_wdir_co
    # print 'N: ', len(high_ywspd)
    #
    # """windsat:"""
    # windsat_file = open('pickle/windsat.pkl', 'rb')
    # windsat = pickle.load(windsat_file)
    # wspdlf_x = windsat['wspd_x']
    # wspdmf_x = windsat['wspd_x']
    # wspdaw_x = windsat['wspd_x']
    # wspdlf_y = windsat['wspdlf_y']
    # wspdmf_y = windsat['wspdmf_y']
    # wspdaw_y = windsat['wspdaw_y']
    # wdir_x = windsat['wdir_x']
    # wdir_y = windsat['wdir_y']
    # # wspdlf_x, wspdlf_y = filt(wspdlf_x, wspdlf_y, limit=20)
    # # wspdmf_x, wspdmf_y = filt(wspdmf_x, wspdmf_y, limit=20)
    # # wspdaw_x, wspdaw_y = filt(wspdaw_x, wspdaw_y, limit=20)
    # # wdir_x, wdir_y = cut_and_change(wdir_x, wdir_y)
    # # wdir_x, wdir_y = change(wdir_x, wdir_y)
    # low_xwspd, low_ywspd, mid1_xwspd, mid1_ywspd, mid2_xwspd, mid2_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid1_xwdir, mid1_ywdir, mid2_xwdir, mid2_ywdir, high_xwdir, high_ywdir = divide2(wspdlf_x, wspdlf_y, wdir_x, wdir_y)
    # low_xwspd, low_ywspd = filt(low_xwspd, low_ywspd, 20)
    # mid1_xwspd, mid1_ywspd = filt(mid1_xwspd, mid1_ywspd, 20)
    # mid2_xwspd, mid2_ywspd = filt(mid2_xwspd, mid2_ywspd, 20)
    # high_xwspd, high_ywspd = filt(high_xwspd, high_ywspd, 20)
    # # low_xwdir, low_ywdir = cut_and_change(low_xwdir, low_ywdir)
    # # mid_xwdir, mid_ywdir = cut_and_change(mid_xwdir, mid_ywdir)
    # # high_xwdir, high_ywdir = cut_and_change(high_xwdir, high_ywdir)
    # low_wspd_mean_bias = cal_mean_bias(low_xwspd, low_ywspd)
    # low_wspd_rmse = cal_rmse(low_xwspd, low_ywspd)
    # low_wspd_co = cal_co(low_xwspd, low_ywspd)
    # mid1_wspd_mean_bias = cal_mean_bias(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_rmse = cal_rmse(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_co = cal_co(mid1_xwspd, mid1_ywspd)
    # mid2_wspd_mean_bias = cal_mean_bias(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_rmse = cal_rmse(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_co = cal_co(mid2_xwspd, mid2_ywspd)
    # high_wspd_mean_bias = cal_mean_bias(high_xwspd, high_ywspd)
    # high_wspd_rmse = cal_rmse(high_xwspd, high_ywspd)
    # high_wspd_co = cal_co(high_xwspd, high_ywspd)
    # # low_wdir_mean_bias = cal_mean_bias(low_xwdir, low_ywdir)
    # # low_wdir_rmse = cal_rmse(low_xwdir, low_ywdir)
    # # low_wdir_co = cal_co(low_xwdir, low_ywdir)
    # # mid_wdir_mean_bias = cal_mean_bias(mid_xwdir, mid_ywdir)
    # # mid_wdir_rmse = cal_rmse(mid_xwdir, mid_ywdir)
    # # mid_wdir_co = cal_co(mid_xwdir, mid_ywdir)
    # # high_wdir_mean_bias = cal_mean_bias(high_xwdir, high_ywdir)
    # # high_wdir_rmse = cal_rmse(high_xwdir, high_ywdir)
    # # high_wdir_co = cal_co(high_xwdir, high_ywdir)
    # print 'windsat-lf: '
    # print 'low:'
    # print 'low_wspd_mean_bias: ', low_wspd_mean_bias
    # print 'low_wspd_rmse: ', low_wspd_rmse
    # print 'low_wspd_co: ', low_wspd_co
    # # print 'low_wdir_mean_bias', low_wdir_mean_bias
    # # print 'low_wdir_rmse: ', low_wdir_rmse
    # # print 'low_wdir_co: ', low_wdir_co
    # print 'N: ', len(low_ywspd)
    # print 'mid1:'
    # print 'mid1_wspd_mean_bias: ', mid1_wspd_mean_bias
    # print 'mid1_wspd_rmse: ', mid1_wspd_rmse
    # print 'mid1_wspd_co: ', mid1_wspd_co
    # # print 'mid_wdir_mean_bias', mid_wdir_mean_bias
    # # print 'mid_wdir_rmse: ', mid_wdir_rmse
    # # print 'mid_wdir_co: ', mid_wdir_co
    # print 'N: ', len(mid1_ywspd)
    # print 'mid2:'
    # print 'mid2_wspd_mean_bias: ', mid2_wspd_mean_bias
    # print 'mid2_wspd_rmse: ', mid2_wspd_rmse
    # print 'mid2_wspd_co: ', mid2_wspd_co
    # print 'N: ', len(mid2_ywspd)
    # print 'high:'
    # print 'high_wspd_mean_bias: ', high_wspd_mean_bias
    # print 'high_wspd_rmse: ', high_wspd_rmse
    # print 'high_wspd_co: ', high_wspd_co
    # # print 'high_wdir_mean_bias', high_wdir_mean_bias
    # # print 'high_wdir_rmse: ', high_wdir_rmse
    # # print 'high_wdir_co: ', high_wdir_co
    # print 'N: ', len(high_ywspd)
    #
    # low_xwspd, low_ywspd, mid1_xwspd, mid1_ywspd, mid2_xwspd, mid2_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid1_xwdir, mid1_ywdir, mid2_xwdir, mid2_ywdir, high_xwdir, high_ywdir = divide2(
    #     wspdmf_x, wspdmf_y, wdir_x, wdir_y)
    # low_xwspd, low_ywspd = filt(low_xwspd, low_ywspd, 20)
    # mid1_xwspd, mid1_ywspd = filt(mid1_xwspd, mid1_ywspd, 20)
    # mid2_xwspd, mid2_ywspd = filt(mid2_xwspd, mid2_ywspd, 20)
    # high_xwspd, high_ywspd = filt(high_xwspd, high_ywspd, 20)
    # # low_xwdir, low_ywdir = cut_and_change(low_xwdir, low_ywdir)
    # # mid_xwdir, mid_ywdir = cut_and_change(mid_xwdir, mid_ywdir)
    # # high_xwdir, high_ywdir = cut_and_change(high_xwdir, high_ywdir)
    # low_wspd_mean_bias = cal_mean_bias(low_xwspd, low_ywspd)
    # low_wspd_rmse = cal_rmse(low_xwspd, low_ywspd)
    # low_wspd_co = cal_co(low_xwspd, low_ywspd)
    # mid1_wspd_mean_bias = cal_mean_bias(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_rmse = cal_rmse(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_co = cal_co(mid1_xwspd, mid1_ywspd)
    # mid2_wspd_mean_bias = cal_mean_bias(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_rmse = cal_rmse(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_co = cal_co(mid2_xwspd, mid2_ywspd)
    # high_wspd_mean_bias = cal_mean_bias(high_xwspd, high_ywspd)
    # high_wspd_rmse = cal_rmse(high_xwspd, high_ywspd)
    # high_wspd_co = cal_co(high_xwspd, high_ywspd)
    # # low_wdir_mean_bias = cal_mean_bias(low_xwdir, low_ywdir)
    # # low_wdir_rmse = cal_rmse(low_xwdir, low_ywdir)
    # # low_wdir_co = cal_co(low_xwdir, low_ywdir)
    # # mid_wdir_mean_bias = cal_mean_bias(mid_xwdir, mid_ywdir)
    # # mid_wdir_rmse = cal_rmse(mid_xwdir, mid_ywdir)
    # # mid_wdir_co = cal_co(mid_xwdir, mid_ywdir)
    # # high_wdir_mean_bias = cal_mean_bias(high_xwdir, high_ywdir)
    # # high_wdir_rmse = cal_rmse(high_xwdir, high_ywdir)
    # # high_wdir_co = cal_co(high_xwdir, high_ywdir)
    # print 'windsat-mf: '
    # print 'low:'
    # print 'low_wspd_mean_bias: ', low_wspd_mean_bias
    # print 'low_wspd_rmse: ', low_wspd_rmse
    # print 'low_wspd_co: ', low_wspd_co
    # # print 'low_wdir_mean_bias', low_wdir_mean_bias
    # # print 'low_wdir_rmse: ', low_wdir_rmse
    # # print 'low_wdir_co: ', low_wdir_co
    # print 'N: ', len(low_ywspd)
    # print 'mid1:'
    # print 'mid1_wspd_mean_bias: ', mid1_wspd_mean_bias
    # print 'mid1_wspd_rmse: ', mid1_wspd_rmse
    # print 'mid1_wspd_co: ', mid1_wspd_co
    # # print 'mid_wdir_mean_bias', mid_wdir_mean_bias
    # # print 'mid_wdir_rmse: ', mid_wdir_rmse
    # # print 'mid_wdir_co: ', mid_wdir_co
    # print 'N: ', len(mid1_ywspd)
    # print 'mid2:'
    # print 'mid2_wspd_mean_bias: ', mid2_wspd_mean_bias
    # print 'mid2_wspd_rmse: ', mid2_wspd_rmse
    # print 'mid2_wspd_co: ', mid2_wspd_co
    # print 'N: ', len(mid2_ywspd)
    # print 'high:'
    # print 'high_wspd_mean_bias: ', high_wspd_mean_bias
    # print 'high_wspd_rmse: ', high_wspd_rmse
    # print 'high_wspd_co: ', high_wspd_co
    # # print 'high_wdir_mean_bias', high_wdir_mean_bias
    # # print 'high_wdir_rmse: ', high_wdir_rmse
    # # print 'high_wdir_co: ', high_wdir_co
    # print 'N: ', len(high_ywspd)
    #
    # low_xwspd, low_ywspd, mid1_xwspd, mid1_ywspd, mid2_xwspd, mid2_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid1_xwdir, mid1_ywdir, mid2_xwdir, mid2_ywdir, high_xwdir, high_ywdir = divide2(
    #     wspdaw_x, wspdaw_y, wdir_x, wdir_y)
    # low_xwspd, low_ywspd = filt(low_xwspd, low_ywspd, 20)
    # mid1_xwspd, mid1_ywspd = filt(mid1_xwspd, mid1_ywspd, 20)
    # mid2_xwspd, mid2_ywspd = filt(mid2_xwspd, mid2_ywspd, 20)
    # high_xwspd, high_ywspd = filt(high_xwspd, high_ywspd, 20)
    # # low_xwdir, low_ywdir = cut_and_change(low_xwdir, low_ywdir)
    # # mid_xwdir, mid_ywdir = cut_and_change(mid_xwdir, mid_ywdir)
    # # high_xwdir, high_ywdir = cut_and_change(high_xwdir, high_ywdir)
    # low_wspd_mean_bias = cal_mean_bias(low_xwspd, low_ywspd)
    # low_wspd_rmse = cal_rmse(low_xwspd, low_ywspd)
    # low_wspd_co = cal_co(low_xwspd, low_ywspd)
    # mid1_wspd_mean_bias = cal_mean_bias(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_rmse = cal_rmse(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_co = cal_co(mid1_xwspd, mid1_ywspd)
    # mid2_wspd_mean_bias = cal_mean_bias(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_rmse = cal_rmse(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_co = cal_co(mid2_xwspd, mid2_ywspd)
    # high_wspd_mean_bias = cal_mean_bias(high_xwspd, high_ywspd)
    # high_wspd_rmse = cal_rmse(high_xwspd, high_ywspd)
    # high_wspd_co = cal_co(high_xwspd, high_ywspd)
    # # low_wdir_mean_bias = cal_mean_bias(low_xwdir, low_ywdir)
    # # low_wdir_rmse = cal_rmse(low_xwdir, low_ywdir)
    # # low_wdir_co = cal_co(low_xwdir, low_ywdir)
    # # mid_wdir_mean_bias = cal_mean_bias(mid_xwdir, mid_ywdir)
    # # mid_wdir_rmse = cal_rmse(mid_xwdir, mid_ywdir)
    # # mid_wdir_co = cal_co(mid_xwdir, mid_ywdir)
    # # high_wdir_mean_bias = cal_mean_bias(high_xwdir, high_ywdir)
    # # high_wdir_rmse = cal_rmse(high_xwdir, high_ywdir)
    # # high_wdir_co = cal_co(high_xwdir, high_ywdir)
    # print 'windsat-aw: '
    # print 'low:'
    # print 'low_wspd_mean_bias: ', low_wspd_mean_bias
    # print 'low_wspd_rmse: ', low_wspd_rmse
    # print 'low_wspd_co: ', low_wspd_co
    # # print 'low_wdir_mean_bias', low_wdir_mean_bias
    # # print 'low_wdir_rmse: ', low_wdir_rmse
    # # print 'low_wdir_co: ', low_wdir_co
    # print 'N: ', len(low_ywspd)
    # print 'mid1:'
    # print 'mid1_wspd_mean_bias: ', mid1_wspd_mean_bias
    # print 'mid1_wspd_rmse: ', mid1_wspd_rmse
    # print 'mid1_wspd_co: ', mid1_wspd_co
    # # print 'mid_wdir_mean_bias', mid_wdir_mean_bias
    # # print 'mid_wdir_rmse: ', mid_wdir_rmse
    # # print 'mid_wdir_co: ', mid_wdir_co
    # print 'N: ', len(mid1_ywspd)
    # print 'mid2:'
    # print 'mid2_wspd_mean_bias: ', mid2_wspd_mean_bias
    # print 'mid2_wspd_rmse: ', mid2_wspd_rmse
    # print 'mid2_wspd_co: ', mid2_wspd_co
    # print 'N: ', len(mid2_ywspd)
    # print 'high:'
    # print 'high_wspd_mean_bias: ', high_wspd_mean_bias
    # print 'high_wspd_rmse: ', high_wspd_rmse
    # print 'high_wspd_co: ', high_wspd_co
    # # print 'high_wdir_mean_bias', high_wdir_mean_bias
    # # print 'high_wdir_rmse: ', high_wdir_rmse
    # # print 'high_wdir_co: ', high_wdir_co
    # print 'N: ', len(high_ywspd)
    #
    # era_file = open('pickle/era.pkl', 'rb')
    # era = pickle.load(era_file)
    # wspd_x = era['wspd_x']
    # wspd_y = era['wspd_y']
    # wdir_x = era['wdir_x']
    # wdir_y = era['wdir_y']
    # low_xwspd, low_ywspd, mid1_xwspd, mid1_ywspd, mid2_xwspd, mid2_ywspd, high_xwspd, high_ywspd, low_xwdir, low_ywdir, mid1_xwdir, mid1_ywdir, mid2_xwdir, mid2_ywdir, high_xwdir, high_ywdir = divide2(
    #     wspd_x, wspd_y, wdir_x, wdir_y)
    # low_xwspd, low_ywspd = filt(low_xwspd, low_ywspd, 20)
    # mid1_xwspd, mid1_ywspd = filt(mid1_xwspd, mid1_ywspd, 20)
    # mid2_xwspd, mid2_ywspd = filt(mid2_xwspd, mid2_ywspd, 20)
    # high_xwspd, high_ywspd = filt(high_xwspd, high_ywspd, 20)
    # # low_xwdir, low_ywdir = cut_and_change(low_xwdir, low_ywdir)
    # # mid_xwdir, mid_ywdir = cut_and_change(mid_xwdir, mid_ywdir)
    # # high_xwdir, high_ywdir = cut_and_change(high_xwdir, high_ywdir)
    # low_wspd_mean_bias = cal_mean_bias(low_xwspd, low_ywspd)
    # low_wspd_rmse = cal_rmse(low_xwspd, low_ywspd)
    # low_wspd_co = cal_co(low_xwspd, low_ywspd)
    # mid1_wspd_mean_bias = cal_mean_bias(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_rmse = cal_rmse(mid1_xwspd, mid1_ywspd)
    # mid1_wspd_co = cal_co(mid1_xwspd, mid1_ywspd)
    # mid2_wspd_mean_bias = cal_mean_bias(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_rmse = cal_rmse(mid2_xwspd, mid2_ywspd)
    # mid2_wspd_co = cal_co(mid2_xwspd, mid2_ywspd)
    # high_wspd_mean_bias = cal_mean_bias(high_xwspd, high_ywspd)
    # high_wspd_rmse = cal_rmse(high_xwspd, high_ywspd)
    # high_wspd_co = cal_co(high_xwspd, high_ywspd)
    # # low_wdir_mean_bias = cal_mean_bias(low_xwdir, low_ywdir)
    # # low_wdir_rmse = cal_rmse(low_xwdir, low_ywdir)
    # # low_wdir_co = cal_co(low_xwdir, low_ywdir)
    # # mid_wdir_mean_bias = cal_mean_bias(mid_xwdir, mid_ywdir)
    # # mid_wdir_rmse = cal_rmse(mid_xwdir, mid_ywdir)
    # # mid_wdir_co = cal_co(mid_xwdir, mid_ywdir)
    # # high_wdir_mean_bias = cal_mean_bias(high_xwdir, high_ywdir)
    # # high_wdir_rmse = cal_rmse(high_xwdir, high_ywdir)
    # # high_wdir_co = cal_co(high_xwdir, high_ywdir)
    # print 'era: '
    # print 'low:'
    # print 'low_wspd_mean_bias: ', low_wspd_mean_bias
    # print 'low_wspd_rmse: ', low_wspd_rmse
    # print 'low_wspd_co: ', low_wspd_co
    # # print 'low_wdir_mean_bias', low_wdir_mean_bias
    # # print 'low_wdir_rmse: ', low_wdir_rmse
    # # print 'low_wdir_co: ', low_wdir_co
    # print 'N: ', len(low_ywspd)
    # print 'mid1:'
    # print 'mid1_wspd_mean_bias: ', mid1_wspd_mean_bias
    # print 'mid1_wspd_rmse: ', mid1_wspd_rmse
    # print 'mid1_wspd_co: ', mid1_wspd_co
    # # print 'mid_wdir_mean_bias', mid_wdir_mean_bias
    # # print 'mid_wdir_rmse: ', mid_wdir_rmse
    # # print 'mid_wdir_co: ', mid_wdir_co
    # print 'N: ', len(mid1_ywspd)
    # print 'mid2:'
    # print 'mid2_wspd_mean_bias: ', mid2_wspd_mean_bias
    # print 'mid2_wspd_rmse: ', mid2_wspd_rmse
    # print 'mid2_wspd_co: ', mid2_wspd_co
    # print 'N: ', len(mid2_ywspd)
    # print 'high:'
    # print 'high_wspd_mean_bias: ', high_wspd_mean_bias
    # print 'high_wspd_rmse: ', high_wspd_rmse
    # print 'high_wspd_co: ', high_wspd_co
    # # print 'high_wdir_mean_bias', high_wdir_mean_bias
    # # print 'high_wdir_rmse: ', high_wdir_rmse
    # # print 'high_wdir_co: ', high_wdir_co
    # print 'N: ', len(high_ywspd)

    """rain contamination"""
    # """qscat"""
    # qscat_file = open('pickle/qscat.pkl', 'rb')
    # qscat = pickle.load(qscat_file)
    # qscat_urain = rain_cut(qscat)
    # wspd_x = qscat_urain['wspd_x']
    # wspd_y = qscat_urain['wspd_y']
    # wdir_x = qscat_urain['wdir_x']
    # wdir_y = qscat_urain['wdir_y']
    # wspd_x, wspd_y = filt(wspd_x, wspd_y, limit=20)
    # wdir_x, wdir_y = cut(wdir_x, wdir_y)
    # wspd_mean_bias = cal_mean_bias(wspd_x, wspd_y)
    # wspd_rmse = cal_rmse(wspd_x, wspd_y)
    # wspd_co = cal_co(wspd_x, wspd_y)
    # wdir_mean_bias = cal_mean_bias(wdir_x, wdir_y)
    # wdir_rmse = cal_rmse(wdir_x, wdir_y)
    # wdir_co = cal_co(wdir_x, wdir_y)
    # wspd_content = 'Mean Bias (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nR: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wspd_x))
    # wdir_content = 'Mean Bias ('+r'$^\circ$'+'): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nRMSE ('+r'$^\circ$'+'): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wdir_x))
    # plot_scatter_wspd4(wspd_x, wspd_y, content=wspd_content, name='qscat_wspd_urain', sate='QSCAT')
    # plot_scatter_wdir4(wdir_x, wdir_y, content=wdir_content, name='qscat_wdir_urain', sate='QSCAT')
    #
    # """ascat"""
    # ascat_file = open('pickle/ascat.pkl', 'rb')
    # ascat = pickle.load(ascat_file)
    # ascat_urain = rain_cut(ascat)
    # wspd_x = ascat_urain['wspd_x']
    # wspd_y = ascat_urain['wspd_y']
    # wdir_x = ascat_urain['wdir_x']
    # wdir_y = ascat_urain['wdir_y']
    # wspd_x, wspd_y = filt(wspd_x, wspd_y, limit=20)
    # wdir_x, wdir_y = cut(wdir_x, wdir_y)
    # wspd_mean_bias = cal_mean_bias(wspd_x, wspd_y)
    # wspd_rmse = cal_rmse(wspd_x, wspd_y)
    # wspd_co = cal_co(wspd_x, wspd_y)
    # wdir_mean_bias = cal_mean_bias(wdir_x, wdir_y)
    # wdir_rmse = cal_rmse(wdir_x, wdir_y)
    # wdir_co = cal_co(wdir_x, wdir_y)
    # wspd_content = 'Mean Bias (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nR: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wspd_x))
    # wdir_content = 'Mean Bias ('+r'$^\circ$'+'): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nRMSE ('+r'$^\circ$'+'): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wdir_x))
    # plot_scatter_wspd4(wspd_x, wspd_y, content=wspd_content, name='ascat_wspd_urain', sate='ASCAT')
    # plot_scatter_wdir4(wdir_x, wdir_y, content=wdir_content, name='ascat_wdir_urain', sate='ASCAT')

    """just test!!!"""
    # ref_name = 'datasets/qscat/20081201.gz'
    # ref = rq.read_qscat(ref_name)
    # month = '01'
    # situs = read_situ(filename='datasets/in-situ/OCEN_SHB_GLB_FTM_SHIP-2008' + month + '.TXT')
    # print 'ship finished.'
    # qscats = batch_read_qscat(month)
    # print 'qscat finished.'
    # ascats = batch_read_ascat(month)
    # print 'ascat finished.'
    # windsats = batch_read_windsat(month)
    # print 'windsat finished.'
    # print 'Start to read ERA: ' + month
    # dataset = rn.read_era(filename='datasets/era/2008' + month + '.nc')
    # eras = rn.cut_map(dataset, ref, month)
    # print 'era finished.'
    # print 'start to match:'
    # num = 0
    # for i in range(31):
    #     match = match_new(situs[i+1], qscats[i+1], ascats[i+1], windsats[i+1], eras[i+1])
    #     num = num + len(match)
    # print 'matched number: ', num
    #
    # month = '02'
    # situs = read_situ(filename='datasets/in-situ/OCEN_SHB_GLB_FTM_SHIP-2008' + month + '.TXT')
    # print 'ship finished.'
    # qscats = batch_read_qscat(month)
    # print 'qscat finished.'
    # ascats = batch_read_ascat(month)
    # print 'ascat finished.'
    # windsats = batch_read_windsat(month)
    # print 'windsat finished.'
    # print 'Start to read ERA: ' + month
    # dataset = rn.read_era(filename='datasets/era/2008' + month + '.nc')
    # eras = rn.cut_map(dataset, ref, month)
    # print 'era finished.'
    # print 'start to match:'
    # num = 0
    # for i in range(29):
    #     match = match_new(situs[i + 1], qscats[i + 1], ascats[i + 1], windsats[i + 1], eras[i + 1])
    #     num = num + len(match)
    # print 'matched number: ', num

    """find all matches!!!!"""
    # month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    # situs_list = {}
    # qscats_list = {}
    # ascats_list = {}
    # windsats_list = {}
    # eras_list = {}
    # ref_name = 'datasets/qscat/20081201.gz'
    # ref = rq.read_qscat(ref_name)
    #
    # match_lists = []
    # for month in month_list:
    #     if month == '02':
    #         days = 29
    #     elif month == '04' or month == '06' or month == '09' or month == '11':
    #         days = 30
    #     else:
    #         days = 31
    #     print 'situ: ' + month
    #     situs = read_situ(filename='datasets/in-situ/OCEN_SHB_GLB_FTM_SHIP-2008' + month + '.TXT')
    #     print 'qscat: ' + month
    #     qscats = batch_read_qscat(month)
    #     print 'ascat: ' + month
    #     ascats = batch_read_ascat(month)
    #     print 'windsat: ' + month
    #     windsats = batch_read_windsat(month)
    #     print 'Start to read ERA: ' + month
    #     dataset = rn.read_era(filename='datasets/era/2008' + month + '.nc')
    #     eras = rn.cut_map(dataset, ref, month)
    #     situs_list[month] = situs
    #     eras_list[month] = eras
    #     situs_list[month] = situs
    #     qscats_list[month] = qscats
    #     ascats_list[month] = ascats
    #     windsats_list[month] = windsats
    #     print 'Data reading ' + month + ' finished!'
    #     print 'Start to match: '+ month
    #     """Start to match"""
    #     for i in range(days):
    #         match_list = match_new(situs_list[month][i + 1], qscats_list[month][i + 1], ascats_list[month][i + 1], windsats_list[month][i + 1], eras_list[month][i + 1])
    #         match_lists = match_lists + match_list
    #     print 'match for '+ month +' finished.'
    #
    # print 'Total Number: ', len(match_lists)
    # match_dataset = {}
    # ship_list = []
    # qscat_list = []
    # ascat_list = []
    # windsat_list = []
    # era_list = []
    # for i in range(len(match_lists)):
    #     ship_list.append(match_lists[i]['ship'])
    #     qscat_list.append(match_lists[i]['qscat'])
    #     ascat_list.append(match_lists[i]['ascat'])
    #     windsat_list.append(match_lists[i]['windsat'])
    #     era_list.append(match_lists[i]['era'])
    # match_dataset['ship'] = ship_list
    # match_dataset['qscat'] = qscat_list
    # match_dataset['ascat'] = ascat_list
    # match_dataset['windsat'] = windsat_list
    # match_dataset['era'] = era_list
    # pickle_file = open('pickle/match.pkl', 'wb')
    # pickle.dump(match_dataset, pickle_file)
    # pickle_file.close()
