import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gaussian_kde
import os
import csv

base_dir = '/Users/zhangdongxiang/PycharmProjects/data4all/match/'
qscat_year_list = ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
ascat_year_list = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
# windsat_year_list = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
windsat_year_list = ['2008']

def plot_scatter_wspd(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 30, 5))
    plt.yticks(range(0, 30, 5))
    plt.text(0, 21, content)
    plt.xlabel('NDBC wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    fig.savefig(base_dir+'figure/'+name+'.eps')
    fig.savefig(base_dir+'figure/'+name+'.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')
    plt.close()

def plot_scatter_wspd_low(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 10, 1))
    plt.yticks(range(0, 6, 1))
    plt.text(0, 5, content)
    plt.xlabel('NDBC wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig(base_dir+'figure/'+name+'.eps')
    fig.savefig(base_dir+'figure/'+name+'.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')

def plot_scatter_wspd_mid(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 16, 2))
    plt.yticks(range(5, 11, 1))
    plt.text(0, 10, content)
    plt.xlabel('NDBC wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig(base_dir+'figure/'+name+'.eps')
    fig.savefig(base_dir+'figure/'+name+'.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')

def plot_scatter_wspd_mid2(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 25, 2))
    plt.yticks(range(10, 16, 1))
    plt.text(0, 15, content)
    plt.xlabel('NDBC wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig(base_dir+'figure/'+name+'.eps')
    fig.savefig(base_dir+'figure/'+name+'.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')

def plot_scatter_wspd_high(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 30, 5))
    plt.yticks(range(15, 30, 2))
    plt.text(0, 25, content)
    plt.xlabel('NDBC wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig(base_dir+'figure/'+name+'.eps')
    fig.savefig(base_dir+'figure/'+name+'.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')

def plot_scatter_wspd_sfmr(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 30, 5))
    plt.yticks(range(15, 30, 2))
    plt.text(1, 23, content)
    plt.xlabel('SFMR wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    # plt.title(name)
    # plt.show()
    fig.savefig(base_dir+'figure/'+name+'.eps')
    fig.savefig(base_dir+'figure/'+name+'.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')

def plot_scatter_wspd_sar(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    fig = plt.figure()
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.scatter(x, y, c=density, marker='.')
    plt.plot(x, x, color='black', linewidth=0.5, alpha = 0.2)
    plt.xticks(range(0, 30, 5))
    plt.yticks(range(0, 30, 5))
    plt.text(1, 19, content)
    plt.xlabel('NDBC wind speed (m/s)')
    plt.ylabel(sate + ' wind speed (m/s)')
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    fig.savefig(base_dir+'figure/'+name+'.eps')
    fig.savefig(base_dir+'figure/'+name+'.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')
    plt.close()

def plot_scatter_wdir(x, y, content, name, sate):
    # regr = linear_model.LinearRegression()
    # regr.fit(np.array(x).reshape(-1, 1), y)
    # a, b = regr.coef_, regr.intercept_

    x_change = [i%360 for i in x]
    y_change = [i%360 for i in y]

    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    fig = plt.figure()
    plt.scatter(x_change, y_change, c=density, marker=".")
    plt.plot(range(361), range(361), color='black', linewidth=0.5, alpha=0.2)
    plt.plot(range(180,361), [(i-180) for i in range(180,361)], color='black', linewidth=0.5, linestyle='--', alpha=0.2)
    plt.plot(range(181), [(i+180) for i in range(181)],color='black', linewidth=0.5, linestyle='--', alpha=0.2)
    # plt.xticks(range(0, 361, 60), ['180', '240', '300', '0', '60', '120', '180'])
    # plt.yticks(range(0, 361, 60), ['180', '240', '300', '0', '60', '120', '180'])
    plt.xticks(range(0, 361, 60))
    plt.yticks(range(0, 361, 60))
    plt.xlabel('NDBC wind direction ('+ r'$^\circ$'+ ')')
    plt.ylabel(sate + ' wind direction ('+ r'$^\circ$'+ ')')
    plt.text(180, 20, content)
    cbar = plt.colorbar()
    cbar.set_label('Density (%)')
    # plt.show()
    fig.savefig(base_dir+'figure/' + name + '.eps')
    fig.savefig(base_dir+'figure/' + name + '.pdf')
    fig.savefig(base_dir + 'figure/' + name + '.png')


def plot_bar(num_list, minus_list, content, name, sate):
    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.bar(x=range(1, 27), height=num_list, width=0.8, bottom=0, color='green')
    ax1.set_yticks(range(0, max(num_list), 200))
    ax1.set_ylabel('Number of data')
    ax1.set_xlabel('NDBC wind speed (m/s)')
    ax2 = ax1.twinx()
    ax2.boxplot(minus_list, sym='')
    ax2.plot(range(28), [0 for n in range(28)], color='gray', linewidth=0.8)
    ax2.set_xticklabels(['0~1', '', '', '', '', '5~6', '', '', '', '', '10~11', '', '', '', '', '15~16', '', '', '', '', '20~21', '', '', '', '', '25~26'])
    ax2.set_yticks(range(-8, 9, 4))
    ax2.set_ylabel('Bias from NDBC (m/s)')
    fig.savefig('./figure/' + name + '.eps')
    fig.savefig('./figure/' + name + '.pdf')

def plot_year_series(year_list, num_list, AE_list, RMSE_list, co_list, name, sate):
    fig = plt.figure()

    ax1 = plt.subplot(111)
    ax1.bar(year_list, height=num_list, width=0.8, bottom=0, color='gray')
    plt.xticks(year_list, rotation=45)  # windsat
    # ax1.set_yticks(range(0, 2*max(num_list), 2000))
    ax1.set_yticks(range(0, 2 * max(num_list), 500))  #windsat
    ax1.set_ylabel('Number of data')
    # ax1.set_xlabel('Year of '+sate)
    ax2 = ax1.twinx()
    # ax2.boxplot(minus_list, sym='')
    ax2.plot(year_list, AE_list, "x-", label="MAE(m/s)", linewidth=0.8)
    ax2.plot(year_list, RMSE_list, "+-", label="RMSE(m/s)", linewidth=0.8)
    ax2.plot(year_list, co_list, "*-", label="CC", linewidth=0.8)
    # ax2.set_xticklabels(
    #     ['0~1', '', '', '', '', '5~6', '', '', '', '', '10~11', '', '', '', '', '15~16', '', '', '', '', '20~21', '',
    #      '', '', '', '25~26'])
    ax2.set_yticks(range(-2, 4))
    ax2.grid()
    # ax2.set_ylabel('Mean Bias(AE) & RMSE from NDBC (m/s)'
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

    fig.savefig(base_dir+'figure/' + name + '.eps')
    fig.savefig(base_dir+'figure/' + name + '.pdf')

def plot_year_series_wdir(year_list, num_list, AE_list, RMSE_list, name, sate):
    fig = plt.figure()

    ax1 = plt.subplot(111)
    ax1.bar(year_list, height=num_list, width=0.8, bottom=0, color='gray')
    plt.xticks(year_list, rotation=45)  # windsat
    # ax1.set_yticks(range(0, 2*max(num_list), 2000))
    ax1.set_yticks(range(0, 2 * max(num_list), 500))  # windsat
    ax1.set_ylabel('Number of data')
    # plt.xlabel('Year of '+sate)
    ax2 = ax1.twinx()
    # ax2.boxplot(minus_list, sym='')
    ax2.plot(year_list, AE_list, "x-", label='MAE('+r'$^\circ$'+')', linewidth=0.8)
    ax2.plot(year_list, RMSE_list, "+-", label='RMSE('+r'$^\circ$'+')', linewidth=0.8)
    # ax2.set_xticklabels(
    #     ['0~1', '', '', '', '', '5~6', '', '', '', '', '10~11', '', '', '', '', '15~16', '', '', '', '', '20~21', '',
    #      '', '', '', '25~26'])
    ax2.set_yticks(range(0, 55, 10))
    ax2.grid()
    # ax2.set_ylabel('Mean Bias(AE) & RMSE from NDBC (m/s)'
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

    fig.savefig(base_dir+'figure/' + name + '.eps')
    fig.savefig(base_dir+'figure/' + name + '.pdf')

def plot_rose(rmse_list, name):
    print(rmse_list)
    rmse_list_plot = rmse_list+[rmse_list[0]]
    rose_x = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi, np.pi * 5 / 4, np.pi * 3 / 2, np.pi*7/4, 0]
    print(rmse_list_plot)
    print(rose_x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    # ax.bar(x=[0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*2], height=rmse_list, width=np.pi/4, color='green')
    ax.plot(rose_x, rmse_list_plot, 'o-', color='green')
    fig.savefig('./figure/' + name + '.eps')
    fig.savefig('./figure/' + name + '.pdf')

def static_num_bias(datasets, minus):
    num_list = [0 for n in range(26)]
    minus_list = [list() for n in range(26)]
    for i in range(len(datasets)):
        if datasets[i] > 25:
            continue
        num_list[int(datasets[i])] += 1
        minus_list[int(datasets[i])].append(minus[i])
    print(num_list)
    print(num_list[16])
    print(len(minus_list[17]))
    return num_list, minus_list

def static_dir_rmse(wdir_datasets, wspd_x, wspd_y):
    num_list = [0 for n in range(8)]
    rmse_list = [0 for n in range(8)]
    x_list = [list() for n in range(8)]
    y_list = [list() for n in range(8)]
    for i in range(len(wdir_datasets)):
        if wdir_datasets[i] > 360:
            continue
        if wdir_datasets[i] == 360:
            wdir_datasets[i] = 0
        num_list[int(wdir_datasets[i]/45)] += 1
        x_list[int(wdir_datasets[i]/45)].append(wspd_x[i])
        y_list[int(wdir_datasets[i]/45)].append(wspd_y[i])
    for i in range(len(num_list)):
        rmse_list[i] = rmse(x_list[i], y_list[i])
    return num_list, rmse_list

def rmse(x_list, y_list):
    rmse_list = list(map(lambda x: (x[1] - x[0]) ** 2, zip(x_list, y_list)))
    # print("%s\n%s\n%s" % (x_list, y_list, rmse_list))
    rmse = math.sqrt(sum(rmse_list) / len(rmse_list))
    return rmse

def mean_bias(x_list, y_list):
    bias_list = list(map(lambda x: x[1] - x[0], zip(x_list, y_list)))
    # print("%s\n%s\n%s" % (x_list, y_list, bias_list))
    bias_mean = sum(bias_list) / len(bias_list)
    return bias_mean

def abs_mean_bias(x_list, y_list):
    bias_list = list(map(lambda x: abs(x[1] - x[0]), zip(x_list, y_list)))
    # print("%s\n%s\n%s" % (x_list, y_list, bias_list))
    bias_mean = sum(bias_list) / len(bias_list)
    return bias_mean

def bias_wdir(x_list, y_list):
    bias_list = list(map(lambda x: x[1] - x[0], zip(x_list, y_list)))
    for i in range(len(bias_list)):
        if bias_list[i] > 180:
            bias_list[i] = 360 - bias_list[i]
        elif bias_list[i] < -180:
            bias_list[i] = 360 + bias_list[i]
    return bias_list

def abs_bias_wdir(x_list, y_list):
    bias_list = list(map(lambda x: abs(x[1] - x[0]), zip(x_list, y_list)))
    for i in range(len(bias_list)):
        if bias_list[i] > 180:
            bias_list[i] = 360 - bias_list[i]
    return bias_list

def rmse_wdir(x_list, y_list):
    bias_list = bias_wdir(x_list, y_list)
    rmse_list = list(map(lambda x: x ** 2, bias_list))
    # print("%s\n%s\n%s" % (x_list, y_list, rmse_list))
    rmse = math.sqrt(sum(rmse_list) / len(rmse_list))
    return rmse

def mean_bias_wdir(x_list, y_list):
    bias_list = bias_wdir(x_list, y_list)
    # print("%s\n%s\n%s" % (x_list, y_list, bias_list))
    bias_mean = sum(bias_list) / len(bias_list)
    return bias_mean

def abs_mean_bias_wdir(x_list, y_list):
    bias_list = abs_bias_wdir(x_list, y_list)
    # print("%s\n%s\n%s" % (x_list, y_list, bias_list))
    bias_mean = sum(bias_list) / len(bias_list)
    return bias_mean

def mean(x):
    return sum(x) / len(x)

def std(x):
    m = mean(x)
    std_list = []
    for i in x:
        std_list.append((i - m)**2)
    if len(std_list) == 1:
        return 0.0000000001
    std = math.sqrt(sum(std_list) / (len(std_list) - 1))
    return std

def co(x_list, y_list):
    x_mean = mean(x_list)
    y_mean = mean(y_list)
    x_std = std(x_list)
    y_std = std(y_list)
    co_list = []
    length = len(x_list)
    for i in range(length):
        co_list.append((x_list[i] - x_mean) * (y_list[i] - y_mean) / (x_std * y_std))
    c = sum(co_list) / len(co_list)
    return c

def filt(wspd_x, wspd_y, wdir_x, wdir_y):
    wspd_x_new = []
    wspd_y_new = []
    wdir_x_new = []
    wdir_y_new = []
    for i in range(len(wspd_x)):
        if wdir_x[i] > 360 or wspd_x[i] > 30:
            continue
        wspd_x_new.append(wspd_x[i])
        wspd_y_new.append(wspd_y[i])
        wdir_x_new.append(wdir_x[i])
        wdir_y_new.append(wdir_y[i])
    return wspd_x_new, wspd_y_new, wdir_x_new, wdir_y_new

def filt_for_rain(wspd_x, wspd_y, wdir_x, wdir_y, rain):
    wspd_x_new = []
    wspd_y_new = []
    wdir_x_new = []
    wdir_y_new = []
    for i in range(len(wspd_x)):
        if not rain[i]:
            continue
        wspd_x_new.append(wspd_x[i])
        wspd_y_new.append(wspd_y[i])
        wdir_x_new.append(wdir_x[i])
        wdir_y_new.append(wdir_y[i])
    return wspd_x_new, wspd_y_new, wdir_x_new, wdir_y_new

def filt_with_rain(wspd_x, wspd_y, wdir_x, wdir_y, rain):
    wspd_x_new = []
    wspd_y_new = []
    wdir_x_new = []
    wdir_y_new = []
    for i in range(len(wspd_x)):
        if not rain[i]:
            wspd_x_new.append(wspd_x[i])
            wspd_y_new.append(wspd_y[i])
            wdir_x_new.append(wdir_x[i])
            wdir_y_new.append(wdir_y[i])
    return wspd_x_new, wspd_y_new, wdir_x_new, wdir_y_new

if __name__ == '__main__':
    """
    Old Style
    """
#     files_ascat = os.listdir('./pickle/ascat')
#     wspds_x = []
#     wspds_y = []
#     wdirs_x = []
#     wdirs_y = []
#     for file in files_ascat:
#         if not os.path.isdir(file):
#             ascat_file = open('./pickle/ascat/' + file, 'rb')
#             ascat = pickle.load(ascat_file)
#             wspd_x = ascat['wspd_ndbc']
#             wspd_y = ascat['wspd_ascat']
#             wdir_x = ascat['wdir_ndbc']
#             wdir_y = ascat['wdir_ascat']
#             for i in range(len(wdir_x)):
#                 if wdir_x[i] >= wdir_y[i]:
#                     wdir_y[i] += 180
#                 else:
#                     wdir_y[i] -= 180
#             wspd_x, wspd_y, wdir_x, wdir_y = filt(wspd_x, wspd_y, wdir_x, wdir_y)
#             wspds_x += wspd_x
#             wspds_y += wspd_y
#             wdirs_x += wdir_x
#             wdirs_y += wdir_y
#     biases_list = list(map(lambda x: x[1] - x[0], zip(wspds_x, wspds_y)))
#
#     wspd_mean_bias = mean_bias(wspds_x, wspds_y)
#     wspd_rmse = rmse(wspds_x, wspds_y)
#     wspd_co = co(wspds_x, wspds_y)
#     wspd_content = 'Mean Bias (m/s): ' + str(round(wspd_mean_bias, 3)) + \
#                     '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
#                    '\nR: ' + str(round(wspd_co, 3)) + \
#                    '\nNumber of Matches: ' + str(len(wspds_x))
#
#     wdir_mean_bias = mean_bias_wdir(wdirs_x, wdirs_y)
#     wdir_rmse = rmse_wdir(wdirs_x, wdirs_y)
#     wdir_co = co(wdirs_x, wdirs_y)
#     wdir_content = 'Mean Bias ('+ r'$^\circ$'+ '): ' + str(round(wdir_mean_bias, 3)) + \
#                    '\nRMSE ('+ r'$^\circ$'+ '): ' + str(round(wdir_rmse, 3)) + \
#                    '\nR: ' + str(round(wdir_co, 3)) + \
#                    '\nNumber of Matches: ' + str(len(wdirs_x))
#
# # plot_scatter_wspd(wspd_x, wspd_y, content=wspd_content, name='ascat_wspd', sate='ASCAT')
# #     plot_scatter_wspd(wspds_x, wspds_y, content=wspd_content, name='ascat_wspd_all', sate='ASCAT')
#     plot_scatter_wdir(wdirs_x, wdirs_y, content=wdir_content, name='ascat_wdir_all', sate='ASCAT')
#
# #     num_list, minus_list = static_num_bias(wspds_x, biases_list)
# #     plot_bar(num_list, minus_list, content='shishishsi2322', name='ascat_bias_all', sate='ASCAT')
# #
# #     num_list, rmse_list = static_dir_rmse(wdirs_x, wspds_x, wspds_y)
# # # plot_rose(rmse_list, name='ascat_rose')
# #     plot_rose(rmse_list, name='ascat_rose_all')

    """
    New Style
    """
    # file = base + filename
    # wspds_x = []
    # wspds_y = []
    # wdirs_x = []
    # wdirs_y = []
    # ascat_file = open(file, 'rb')
    # ascats = pickle.load(ascat_file)
    # for ascat in ascats:
    #     wspds_x.append(ascat['a_wspd'])
    #     wspds_y.append(ascat['b_wspd'])
    #     wdirs_x.append(ascat['a_wdir'])
    #     wdirs_y.append(ascat['b_wdir'])
    # for i in range(len(wdirs_x)):
    #     if wdirs_x[i] >= wdirs_y[i]:
    #         wdirs_y[i] += 180
    #     else:
    #         wdirs_y[i] -= 180
    # wspds_x, wspds_y, wdirs_x, wdirs_y = filt(wspds_x, wspds_y, wdirs_x, wdirs_y)
    # biases_list = list(map(lambda x: x[1] - x[0], zip(wspds_x, wspds_y)))
    #
    # wspd_mean_bias = mean_bias(wspds_x, wspds_y)
    # wspd_rmse = rmse(wspds_x, wspds_y)
    # wspd_co = co(wspds_x, wspds_y)
    # wspd_content = 'Mean Bias (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nR: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wspds_x))
    #
    # wdir_mean_bias = mean_bias_wdir(wdirs_x, wdirs_y)
    # wdir_rmse = rmse_wdir(wdirs_x, wdirs_y)
    # wdir_co = co(wdirs_x, wdirs_y)
    # wdir_content = 'Mean Bias ('+ r'$^\circ$'+ '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nRMSE ('+ r'$^\circ$'+ '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(wdirs_x))
    #
    # plot_scatter_wspd(wspds_x, wspds_y, content=wspd_content, name='ascat_wspd_new', sate='ASCAT')
    # plot_scatter_wdir(wdirs_x, wdirs_y, content=wdir_content, name='ascat_wdir_new', sate='ASCAT')
    
    #     num_list, minus_list = static_num_bias(wspds_x, biases_list)
    #     plot_bar(num_list, minus_list, content='shishishsi2322', name='ascat_bias_all', sate='ASCAT')
    # 
    #     num_list, rmse_list = static_dir_rmse(wdirs_x, wspds_x, wspds_y)
    # # plot_rose(rmse_list, name='ascat_rose')
    #     plot_rose(rmse_list, name='ascat_rose_all')

    """
    all time validation
    """
    """qscat"""
    # ndbc_wspds = []
    # windsat_wspds = []
    # ndbc_wdirs = []
    # windsat_wdirs = []
    # for year in windsat_year_list:
    #     dir = base_dir + 'windsat/' + year
    #     files = os.listdir(dir)
    #     files.sort()
    #     for file in files:
    #         if file[5] != '_':
    #             continue
    #         windsat_file = open(dir+'/'+file, 'rb')
    #         windsat = pickle.load(windsat_file)
    #         for point in windsat:
    #             ndbc_wspd = point['a_wspd']
    #             windsat_wspd = point['b_wspd']
    #             ndbc_wdir = point['a_wdir']
    #             windsat_wdir = point['b_wdir']
    #             if ndbc_wdir >= windsat_wdir:
    #                 windsat_wdir += 180
    #             else:
    #                 windsat_wdir -= 180
    #             ndbc_wspds.append(ndbc_wspd)
    #             windsat_wspds.append(windsat_wspd)
    #             ndbc_wdirs.append(ndbc_wdir)
    #             windsat_wdirs.append(windsat_wdir)
    #
    # ndbc_wspds, windsat_wspds, ndbc_wdirs, windsat_wdirs = filt(ndbc_wspds, windsat_wspds, ndbc_wdirs, windsat_wdirs)
    #
    # wspd_mean_bias = mean_bias(ndbc_wspds, windsat_wspds)
    # wspd_abs_mean_bias = abs_mean_bias(ndbc_wspds, windsat_wspds)
    # wspd_rmse = rmse(ndbc_wspds, windsat_wspds)
    # wspd_co = co(ndbc_wspds, windsat_wspds)
    # wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nR: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(ndbc_wspds))
    #
    # wdir_mean_bias = mean_bias_wdir(ndbc_wdirs, windsat_wdirs)
    # wdir_abs_mean_bias = abs_mean_bias_wdir(ndbc_wdirs, windsat_wdirs)
    # wdir_rmse = rmse_wdir(ndbc_wdirs, windsat_wdirs)
    # wdir_co = co(ndbc_wdirs, windsat_wdirs)
    # wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(ndbc_wdirs))
    #
    # plot_scatter_wspd(ndbc_wspds, windsat_wspds, content=wspd_content, name='windsat_wspd_for_all_year', sate='WindSat')
    # plot_scatter_wdir(ndbc_wdirs, windsat_wdirs, content=wdir_content, name='windsat_wdir_for_all_year', sate='WindSat')
    # 

    
    """
    validation in diff wspd 
    """
    """qscat"""
    # ndbc_wspds = []
    # qscat_wspds = []
    # ndbc_wdirs = []
    # qscat_wdirs = []
    # for year in qscat_year_list:
    #     dir = base_dir + 'qscat/' + year
    #     files = os.listdir(dir)
    #     files.sort()
    #     for file in files:
    #         if file[5] != '_':
    #             continue
    #         qscat_file = open(dir + '/' + file, 'rb')
    #         qscat = pickle.load(qscat_file)
    #         for point in qscat:
    #             ndbc_wspd = point['a_wspd']
    #             qscat_wspd = point['b_wspd']
    #             ndbc_wdir = point['a_wdir']
    #             qscat_wdir = point['b_wdir']
    #             if ndbc_wdir >= qscat_wdir:
    #                 qscat_wdir += 180
    #             else:
    #                 qscat_wdir -= 180
    #             ndbc_wspds.append(ndbc_wspd)
    #             qscat_wspds.append(qscat_wspd)
    #             ndbc_wdirs.append(ndbc_wdir)
    #             qscat_wdirs.append(qscat_wdir)
    #
    # ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs = filt(ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs)
    # low_ndbc_wspds = []
    # low_qscat_wspds = []
    # low_ndbc_wdirs = []
    # low_qscat_wdirs = []
    # mid_ndbc_wspds = []
    # mid_qscat_wspds = []
    # mid_ndbc_wdirs = []
    # mid_qscat_wdirs = []
    # mid2_ndbc_wspds = []
    # mid2_qscat_wspds = []
    # mid2_ndbc_wdirs = []
    # mid2_qscat_wdirs = []
    # high_ndbc_wspds = []
    # high_qscat_wspds = []
    # high_ndbc_wdirs = []
    # high_qscat_wdirs = []
    # for i in range(len(qscat_wspds)):
    #     if qscat_wspds[i] < 5:
    #         low_ndbc_wspds.append(ndbc_wspds[i])
    #         low_qscat_wspds.append(qscat_wspds[i])
    #         low_ndbc_wdirs.append(ndbc_wdirs[i])
    #         low_qscat_wdirs.append(qscat_wdirs[i])
    #     elif qscat_wspds[i] < 10:
    #         mid_ndbc_wspds.append(ndbc_wspds[i])
    #         mid_qscat_wspds.append(qscat_wspds[i])
    #         mid_ndbc_wdirs.append(ndbc_wdirs[i])
    #         mid_qscat_wdirs.append(qscat_wdirs[i])
    #     elif qscat_wspds[i] < 15:
    #         mid2_ndbc_wspds.append(ndbc_wspds[i])
    #         mid2_qscat_wspds.append(qscat_wspds[i])
    #         mid2_ndbc_wdirs.append(ndbc_wdirs[i])
    #         mid2_qscat_wdirs.append(qscat_wdirs[i])
    #     else:
    #         high_ndbc_wspds.append(ndbc_wspds[i])
    #         high_qscat_wspds.append(qscat_wspds[i])
    #         high_ndbc_wdirs.append(ndbc_wdirs[i])
    #         high_qscat_wdirs.append(qscat_wdirs[i])
    #
    # wspd_abs_mean_bias = abs_mean_bias(low_ndbc_wspds, low_qscat_wspds)
    # wspd_mean_bias = mean_bias(low_ndbc_wspds, low_qscat_wspds)
    # wspd_rmse = rmse(low_ndbc_wspds, low_qscat_wspds)
    # wspd_co = co(low_ndbc_wspds, low_qscat_wspds)
    # wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nCC: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(low_ndbc_wspds))
    # print("qscat wspd in low:")
    # print(wspd_content)
    #
    # wdir_abs_mean_bias = abs_mean_bias_wdir(low_ndbc_wdirs, low_qscat_wdirs)
    # wdir_mean_bias = mean_bias_wdir(low_ndbc_wdirs, low_qscat_wdirs)
    # wdir_rmse = rmse_wdir(low_ndbc_wdirs, low_qscat_wdirs)
    # wdir_co = co(low_ndbc_wdirs, low_qscat_wdirs)
    # wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(low_ndbc_wdirs))
    # print("qscat wdir in low:")
    # print(wdir_content)
    #
    # plot_scatter_wspd_low(low_ndbc_wspds, low_qscat_wspds, content=wspd_content, name='qscat_wspd_for_low',
    #                       sate='QSCAT')
    # plot_scatter_wdir(low_ndbc_wdirs, low_qscat_wdirs, content=wdir_content, name='qscat_wdir_for_low',
    #                   sate='QSCAT')
    #
    # wspd_abs_mean_bias = abs_mean_bias(mid_ndbc_wspds, mid_qscat_wspds)
    # wspd_mean_bias = mean_bias(mid_ndbc_wspds, mid_qscat_wspds)
    # wspd_rmse = rmse(mid_ndbc_wspds, mid_qscat_wspds)
    # wspd_co = co(mid_ndbc_wspds, mid_qscat_wspds)
    # wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nCC: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(mid_ndbc_wspds))
    # print("qscat wspd in mid:")
    # print(wspd_content)
    #
    # wdir_abs_mean_bias = abs_mean_bias_wdir(mid_ndbc_wdirs, mid_qscat_wdirs)
    # wdir_mean_bias = mean_bias_wdir(mid_ndbc_wdirs, mid_qscat_wdirs)
    # wdir_rmse = rmse_wdir(mid_ndbc_wdirs, mid_qscat_wdirs)
    # wdir_co = co(mid_ndbc_wdirs, mid_qscat_wdirs)
    # wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(mid_ndbc_wdirs))
    # print("qscat wdir in mid:")
    # print(wdir_content)
    #
    # plot_scatter_wspd_mid(mid_ndbc_wspds, mid_qscat_wspds, content=wspd_content, name='qscat_wspd_for_mid',
    #                       sate='QSCAT')
    # plot_scatter_wdir(mid_ndbc_wdirs, mid_qscat_wdirs, content=wdir_content, name='qscat_wdir_for_mid',
    #                   sate='QSCAT')
    #
    # wspd_abs_mean_bias = abs_mean_bias(mid2_ndbc_wspds, mid2_qscat_wspds)
    # wspd_mean_bias = mean_bias(mid2_ndbc_wspds, mid2_qscat_wspds)
    # wspd_rmse = rmse(mid2_ndbc_wspds, mid2_qscat_wspds)
    # wspd_co = co(mid2_ndbc_wspds, mid2_qscat_wspds)
    # wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nCC: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(mid2_ndbc_wspds))
    # print("qscat wspd in mid2:")
    # print(wspd_content)
    #
    # wdir_abs_mean_bias = abs_mean_bias_wdir(mid2_ndbc_wdirs, mid2_qscat_wdirs)
    # wdir_mean_bias = mean_bias(mid2_ndbc_wdirs, mid2_qscat_wdirs)
    # wdir_rmse = rmse_wdir(mid2_ndbc_wdirs, mid2_qscat_wdirs)
    # wdir_co = co(mid2_ndbc_wdirs, mid2_qscat_wdirs)
    # wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(mid2_ndbc_wdirs))
    # print("qscat wdir in mid2:")
    # print(wdir_content)
    #
    # plot_scatter_wspd_mid2(mid2_ndbc_wspds, mid2_qscat_wspds, content=wspd_content, name='qscat_wspd_for_mid2',
    #                        sate='QSCAT')
    # plot_scatter_wdir(mid2_ndbc_wdirs, mid2_qscat_wdirs, content=wdir_content, name='qscat_wdir_for_mid2',
    #                   sate='QSCAT')
    #
    # wspd_abs_mean_bias = abs_mean_bias(high_ndbc_wspds, high_qscat_wspds)
    # wspd_mean_bias = mean_bias(high_ndbc_wspds, high_qscat_wspds)
    # wspd_rmse = rmse(high_ndbc_wspds, high_qscat_wspds)
    # wspd_co = co(high_ndbc_wspds, high_qscat_wspds)
    # wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nCC: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(high_ndbc_wspds))
    # print("qscat wspd in high:")
    # print(wspd_content)
    #
    # wdir_abs_mean_bias = abs_mean_bias_wdir(high_ndbc_wdirs, high_qscat_wdirs)
    # wdir_mean_bias = mean_bias(high_ndbc_wdirs, high_qscat_wdirs)
    # wdir_rmse = rmse_wdir(high_ndbc_wdirs, high_qscat_wdirs)
    # wdir_co = co(high_ndbc_wdirs, high_qscat_wdirs)
    # wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(high_ndbc_wdirs))
    # print("qscat wdir in high:")
    # print(wdir_content)
    #
    # plot_scatter_wspd_high(high_ndbc_wspds, high_qscat_wspds, content=wspd_content, name='qscat_wspd_for_high',
    #                        sate='QSCAT')
    # plot_scatter_wdir(high_ndbc_wdirs, high_qscat_wdirs, content=wdir_content, name='qscat_wdir_for_high',
    #                   sate='QSCAT')


    """
    rain
    """
    """qscat"""
    # ndbc_wspds = []
    # qscat_wspds = []
    # ndbc_wdirs = []
    # qscat_wdirs = []
    # rains = []
    # for year in qscat_year_list:
    #     dir = base_dir + 'qscat/' + year
    #     files = os.listdir(dir)
    #     files.sort()
    #     for file in files:
    #         if file[5] != '_':
    #             continue
    #         qscat_file = open(dir + '/' + file, 'rb')
    #         qscat = pickle.load(qscat_file)
    #         for point in qscat:
    #             ndbc_wspd = point['a_wspd']
    #             qscat_wspd = point['b_wspd']
    #             ndbc_wdir = point['a_wdir']
    #             qscat_wdir = point['b_wdir']
    #             rain = point['rain']
    #             if ndbc_wdir >= qscat_wdir:
    #                 qscat_wdir += 180
    #             else:
    #                 qscat_wdir -= 180
    #             ndbc_wspds.append(ndbc_wspd)
    #             qscat_wspds.append(qscat_wspd)
    #             ndbc_wdirs.append(ndbc_wdir)
    #             qscat_wdirs.append(qscat_wdir)
    #             rains.append(rain)
    #
    # ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs = filt(ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs)
    # ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs = filt_for_rain(ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs, rains)
    # # ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs = filt_with_rain(ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs, rains)
    #
    # wspd_mean_bias = mean_bias(ndbc_wspds, qscat_wspds)
    # wspd_abs_mean_bias = abs_mean_bias(ndbc_wspds, qscat_wspds)
    # wspd_rmse = rmse(ndbc_wspds, qscat_wspds)
    # wspd_co = co(ndbc_wspds, qscat_wspds)
    # wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nR: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(ndbc_wspds))
    #
    # wdir_mean_bias = mean_bias(ndbc_wdirs, qscat_wdirs)
    # wdir_abs_mean_bias = abs_mean_bias_wdir(ndbc_wdirs, qscat_wdirs)
    # wdir_rmse = rmse_wdir(ndbc_wdirs, qscat_wdirs)
    # wdir_co = co(ndbc_wdirs, qscat_wdirs)
    # wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nR: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(ndbc_wdirs))
    #
    # print(wspd_content)
    # print(wdir_content)
    #
    # # plot_scatter_wspd(ndbc_wspds, qscat_wspds, content=wspd_content, name='qscat_wspd_in_rain', sate='QSCAT')
    # # plot_scatter_wdir(ndbc_wdirs, qscat_wdirs, content=wdir_content, name='qscat_wdir_in_rain', sate='QSCAT')



    """
    time analysis
    """
    """windsat"""
    wspd_mean_bias_list = []
    wspd_rmse_list = []
    wspd_co_list = []
    wspd_num_list = []
    wdir_mean_bias_list = []
    wdir_rmse_list = []
    wdir_co_list = []
    wdir_num_list = []

    ndbc_wspds = []
    ascat_wspds = []
    ndbc_wdirs = []
    ascat_wdirs = []
    for year in ['2012', '2017']:
        
        dir = base_dir + 'ascat/' + year
        files = os.listdir(dir)
        files.sort()
        for file in files:
            if file[5] != '_':
                continue
            ascat_file = open(dir + '/' + file, 'rb')
            ascat = pickle.load(ascat_file)
            for point in ascat:
                ndbc_wspd = point['a_wspd']
                ascat_wspd = point['b_wspd']
                ndbc_wdir = point['a_wdir']
                ascat_wdir = point['b_wdir']
                if ndbc_wdir >= ascat_wdir:
                    ascat_wdir += 180
                else:
                    ascat_wdir -= 180
                ndbc_wspds.append(ndbc_wspd)
                ascat_wspds.append(ascat_wspd)
                ndbc_wdirs.append(ndbc_wdir)
                ascat_wdirs.append(ascat_wdir)

    ndbc_wspds, ascat_wspds, ndbc_wdirs, ascat_wdirs = filt(ndbc_wspds, ascat_wspds, ndbc_wdirs, ascat_wdirs)

    wspd_mean_bias = mean_bias(ndbc_wspds, ascat_wspds)
    wspd_abs_mean_bias = abs_mean_bias(ndbc_wspds, ascat_wspds)
    wspd_rmse = rmse(ndbc_wspds, ascat_wspds)
    wspd_co = co(ndbc_wspds, ascat_wspds)
    wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
                   '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
                   '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
                   '\nR: ' + str(round(wspd_co, 3)) + \
                   '\nNumber of Matches: ' + str(len(ndbc_wspds))
    wspd_mean_bias_list.append(wspd_abs_mean_bias)
    wspd_rmse_list.append(wspd_rmse)
    wspd_co_list.append(wspd_co)
    wspd_num_list.append(len(ndbc_wspds))

    wdir_mean_bias = mean_bias_wdir(ndbc_wdirs, ascat_wdirs)
    wdir_abs_mean_bias = abs_mean_bias_wdir(ndbc_wdirs, ascat_wdirs)
    wdir_rmse = rmse_wdir(ndbc_wdirs, ascat_wdirs)
    wdir_co = co(ndbc_wdirs, ascat_wdirs)
    wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
                   '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
                   '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
                   '\nR: ' + str(round(wdir_co, 3)) + \
                   '\nNumber of Matches: ' + str(len(ndbc_wdirs))
    wdir_mean_bias_list.append(wdir_abs_mean_bias)
    wdir_rmse_list.append(wdir_rmse)
    wdir_co_list.append(wdir_co)
    wdir_num_list.append(len(ndbc_wdirs))

        # content = [str(len(ndbc_wspds)), wspd_abs_mean_bias, wspd_rmse, wspd_co, wdir_abs_mean_bias, wdir_rmse, wdir_co]
        # csvFile = open('./ascat.csv', 'a', newline='')
        # writer = csv.writer(csvFile)
        # writer.writerow(content)

        # plot_scatter_wspd(ndbc_wspds, ascat_wspds, content=wspd_content, name='ascat/'+year+'/ascat_wspd_for_'+year, sate='QSCAT')
        # plot_scatter_wdir(ndbc_wdirs, ascat_wdirs, content=wdir_content, name='ascat/'+year+'/ascat_wdir_for_'+year, sate='QSCAT')
    # plot_year_series(ascat_year_list, wspd_num_list,wspd_mean_bias_list,wspd_rmse_list,wspd_co_list,name="time_series_ascat_wspd",sate="QSCAT")
    # plot_year_series_wdir(ascat_year_list, wdir_num_list, wdir_mean_bias_list, wdir_rmse_list,name="time_series_ascat_wdir", sate="QSCAT")

    plot_scatter_wspd(ndbc_wspds, ascat_wspds, content=wspd_content, name='fake_wspd_for_stw',
                      sate='STW Fused Wind Field')
    plot_scatter_wdir(ndbc_wdirs, ascat_wdirs, content=wdir_content, name='fake_wdir_for_stw',
                      sate='STW Fused Wind Field')

    """
    validate by buoys
    """
    """qscat"""
    # buoy_list = []
    # ndbc_wspds_dict = {}
    # ndbc_wdirs_dict = {}
    # qscat_wspds_dict = {}
    # qscat_wdirs_dict = {}
    # for year in qscat_year_list:
    #     dir = base_dir + 'qscat/' + year
    #     files = os.listdir(dir)
    #     files.sort()
    #     for file in files:
    #         if file[5] != '_':
    #             continue
    #         id = file[0:5]
    #         qscat_file = open(dir + '/' + file, 'rb')
    #         qscat = pickle.load(qscat_file)
    #         ndbc_wspds = []
    #         qscat_wspds = []
    #         ndbc_wdirs = []
    #         qscat_wdirs = []
    #         for point in qscat:
    #             ndbc_wspd = point['a_wspd']
    #             qscat_wspd = point['b_wspd']
    #             ndbc_wdir = point['a_wdir']
    #             qscat_wdir = point['b_wdir']
    #             if ndbc_wdir >= qscat_wdir:
    #                 qscat_wdir += 180
    #             else:
    #                 qscat_wdir -= 180
    #             ndbc_wspds.append(ndbc_wspd)
    #             qscat_wspds.append(qscat_wspd)
    #             ndbc_wdirs.append(ndbc_wdir)
    #             qscat_wdirs.append(qscat_wdir)
    #             ndbc_wspds, qscat_wspds, ndbc_wdirs, qscat_wdirs = filt(ndbc_wspds, qscat_wspds, ndbc_wdirs,
    #                                                                     qscat_wdirs)
    #         if id in buoy_list:
    #             ndbc_wspds_dict[id] += ndbc_wspds
    #             qscat_wspds_dict[id] += qscat_wspds
    #             ndbc_wdirs_dict[id] += ndbc_wdirs
    #             qscat_wdirs_dict[id] += qscat_wdirs
    #         else:
    #             ndbc_wspds_dict[id] = ndbc_wspds
    #             qscat_wspds_dict[id] = qscat_wspds
    #             ndbc_wdirs_dict[id] = ndbc_wdirs
    #             qscat_wdirs_dict[id] = qscat_wdirs
    #             buoy_list.append(id)
    #
    # for buoy in buoy_list:
    #     print(buoy)
    #     if not len(ndbc_wspds_dict[buoy]):
    #         continue
    #     wspd_mean_bias = mean_bias(ndbc_wspds_dict[buoy], qscat_wspds_dict[buoy])
    #     wspd_abs_mean_bias = abs_mean_bias(ndbc_wspds_dict[buoy], qscat_wspds_dict[buoy])
    #     wspd_rmse = rmse(ndbc_wspds_dict[buoy], qscat_wspds_dict[buoy])
    #     wspd_co = co(ndbc_wspds_dict[buoy], qscat_wspds_dict[buoy])
    #     wspd_content = 'MBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                    '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                    '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                    '\nR: ' + str(round(wspd_co, 3)) + \
    #                    '\nNumber of Matches: ' + str(len(ndbc_wspds_dict[buoy]))
    #     wdir_mean_bias = mean_bias_wdir(ndbc_wdirs_dict[buoy], qscat_wdirs_dict[buoy])
    #     wdir_abs_mean_bias = abs_mean_bias_wdir(ndbc_wdirs_dict[buoy], qscat_wdirs_dict[buoy])
    #     wdir_rmse = rmse_wdir(ndbc_wdirs_dict[buoy], qscat_wdirs_dict[buoy])
    #     wdir_co = co(ndbc_wdirs_dict[buoy], qscat_wdirs_dict[buoy])
    #     wdir_content = 'MBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                    '\nMAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                    '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                    '\nR: ' + str(round(wdir_co, 3)) + \
    #                    '\nNumber of Matches: ' + str(len(ndbc_wdirs_dict[buoy]))
    #
    #     csvFile = open('/Users/zhangdongxiang/PycharmProjects/data4all/match/distance_qscat.csv', 'a', newline='')
    #     writer = csv.writer(csvFile)
    #     content1 = [buoy, str(round(wspd_mean_bias, 3)), str(round(wspd_abs_mean_bias, 3)), str(round(wspd_rmse, 3)), str(round(wspd_co, 3))]
    #     content2 = [buoy, str(round(wdir_mean_bias, 3)), str(round(wdir_abs_mean_bias, 3)), str(round(wdir_rmse, 3)),
    #                str(round(wdir_co, 3))]
    #     writer.writerow(content1+content2)
    #     csvFile.close()
    #
    #     plot_scatter_wspd(ndbc_wspds_dict[buoy], qscat_wspds_dict[buoy], content=wspd_content,
    #                       name='qscat/qscat_wspd_for_' + buoy, sate='QSCAT')
    #     plot_scatter_wdir(ndbc_wdirs_dict[buoy], qscat_wdirs_dict[buoy], content=wdir_content,
    #                       name='qscat/qscat_wdir_for_' + buoy, sate='QSCAT')


    """
    fade with SFMR
    """
    # buoy_list = []
    # ndbc_wspds_dict = {}
    # ndbc_wdirs_dict = {}
    # ascat_wspds_dict = {}
    # ascat_wdirs_dict = {}
    # for year in ascat_year_list:
    #     dir = base_dir + 'ascat/' + year
    #     files = os.listdir(dir)
    #     files.sort()
    #     for file in files:
    #         if file[5] != '_':
    #             continue
    #         id = file[0:5]
    #         ascat_file = open(dir + '/' + file, 'rb')
    #         ascat = pickle.load(ascat_file)
    #         ndbc_wspds = []
    #         ascat_wspds = []
    #         ndbc_wdirs = []
    #         ascat_wdirs = []
    #         for point in ascat:
    #             ndbc_wspd = point['a_wspd']
    #             ascat_wspd = point['b_wspd']
    #             ndbc_wdir = point['a_wdir']
    #             ascat_wdir = point['b_wdir']
    #             if ndbc_wdir >= ascat_wdir:
    #                 ascat_wdir += 180
    #             else:
    #                 ascat_wdir -= 180
    #             ndbc_wspds.append(ndbc_wspd)
    #             ascat_wspds.append(ascat_wspd)
    #             ndbc_wdirs.append(ndbc_wdir)
    #             ascat_wdirs.append(ascat_wdir)
    #             ndbc_wspds, ascat_wspds, ndbc_wdirs, ascat_wdirs = filt(ndbc_wspds, ascat_wspds, ndbc_wdirs,
    #                                                                     ascat_wdirs)
    #         if id in buoy_list:
    #             ndbc_wspds_dict[id] += ndbc_wspds
    #             ascat_wspds_dict[id] += ascat_wspds
    #             ndbc_wdirs_dict[id] += ndbc_wdirs
    #             ascat_wdirs_dict[id] += ascat_wdirs
    #         else:
    #             ndbc_wspds_dict[id] = ndbc_wspds
    #             ascat_wspds_dict[id] = ascat_wspds
    #             ndbc_wdirs_dict[id] = ndbc_wdirs
    #             ascat_wdirs_dict[id] = ascat_wdirs
    #             buoy_list.append(id)
    #
    # ndbc_wspd_plot = []
    # ascat_wspd_plot = []
    # ndbc_wdir_plot = []
    # ascat_wdir_plot = []
    # for buoy in ['46002', '46005', '46006', '46059']:
    #     print(buoy)
    #     if not len(ndbc_wspds_dict[buoy]):
    #         continue
    #     ndbc_wspd_plot += ndbc_wspds_dict[buoy]
    #     ascat_wspd_plot += ascat_wspds_dict[buoy]
    #     ndbc_wdir_plot += ndbc_wdirs_dict[buoy]
    #     ascat_wdir_plot += ascat_wdirs_dict[buoy]
    #
    # high_ndbc_wspds = []
    # high_ascat_wspds = []
    # high_ndbc_wdirs = []
    # high_ascat_wdirs = []
    # for i in range(len(ascat_wspd_plot)):
    #     if ascat_wspd_plot[i] >= 15:
    #         high_ndbc_wspds.append(ndbc_wspd_plot[i])
    #         high_ascat_wspds.append(ascat_wspd_plot[i])
    #         high_ndbc_wdirs.append(ndbc_wdir_plot[i])
    #         high_ascat_wdirs.append(ascat_wdir_plot[i])
    #
    #
    # wspd_mean_bias = mean_bias(high_ndbc_wspds, high_ascat_wspds)
    # wspd_abs_mean_bias = abs_mean_bias(high_ndbc_wspds, high_ascat_wspds)
    # wspd_rmse = rmse(high_ndbc_wspds, high_ascat_wspds)
    # wspd_co = co(high_ndbc_wspds, high_ascat_wspds)
    # wspd_content = 'MAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nMBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nCC: ' + str(round(wspd_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(high_ascat_wspds))
    #
    # wdir_mean_bias = mean_bias_wdir(high_ndbc_wdirs, high_ascat_wdirs)
    # wdir_abs_mean_bias = abs_mean_bias_wdir(high_ndbc_wdirs, high_ascat_wdirs)
    # wdir_rmse = rmse_wdir(high_ndbc_wdirs, high_ascat_wdirs)
    # wdir_co = co(high_ndbc_wdirs, high_ascat_wdirs)
    # wdir_content = 'MAE (' + r'$^\circ$' + '): ' + str(round(wdir_abs_mean_bias, 3)) + \
    #                '\nMBE (' + r'$^\circ$' + '): ' + str(round(wdir_mean_bias, 3)) + \
    #                '\nRMSE (' + r'$^\circ$' + '): ' + str(round(wdir_rmse, 3)) + \
    #                '\nCC: ' + str(round(wdir_co, 3)) + \
    #                '\nNumber of Matches: ' + str(len(high_ascat_wdirs))
    #
    # plot_scatter_wspd_sfmr(high_ndbc_wspds, high_ascat_wspds, content=wspd_content,
    #                   name='ascat/ascat_wspd_for_sfmr', sate='WindSat')
    # plot_scatter_wdir(high_ndbc_wdirs, high_ascat_wdirs, content=wdir_content,
    #                   name='ascat/ascat_wdir_for_sfmr', sate='WindSat')

    """
    SAR validation
    """
    # envisat = [4.3517313,2.5073109,3.3450356,3.6682246,8.899915,6.0308976,5.307231,1.7458584,5.353388,1.4876592,7.183959,4.656567,3.3673964,5.4229712,5.9118423,7.524879,18.741123,4.1401362,3.211941,7.0067525,7.079337,9.6113,12.570765,7.49084,17.786856,3.856624,6.264511,3.9414332,7.4144397,5.5268397,9.576644,4.9921265,9.228261,11.585407,10.767011,9.284884,9.559994,5.146293,0.4667496,8.87521,12.815211,3.7650828,5.588597,10.638467,7.0333667,4.4279027,12.279236,7.9678845,10.744876]
    # envisat_ndbc = [3.706690954,3.052569021,3.379629988,2.071386122,6.541219331,4.142772243,4.46983321,2.834528377,3.379629988,3.161589343,6.541219331,4.033751921,4.251792565,4.6,5.123955143,5.99611772,5.492605063,5.01493482,3.48865031,9.247484148,7.631422553,10.14967772,13.03653873,6.323178687,3.696945716,1.090203222,5.01493482,3.052569021,9.360258345,7.304361586,10.26245192,7.631422553,9.134709951,10.26245192,10.03690353,9.473032541,9.021935754,4.46983321,2.071386122,7.098349178,12.40516166,3.161589343,3.597670632,9.698580935,6.868280298,3.48865031,14.5478714,5.451016109,9.473032541]
    #
    # wspd_mean_bias = mean_bias(envisat_ndbc, envisat)
    # wspd_abs_mean_bias = abs_mean_bias(envisat_ndbc, envisat)
    # wspd_rmse = rmse(envisat_ndbc, envisat)
    # wspd_co = co(envisat_ndbc, envisat)
    # wspd_content = 'Number of Matches: ' + str(len(envisat)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nMBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nCC: ' + str(round(wspd_co, 3))
    # print("Envisat:\n", wspd_content)
    # plot_scatter_wspd_sar(envisat_ndbc, envisat, content=wspd_content, name='sar/envisat_wspd', sate='Envisat')
    #
    # radarsat = [22.929369,7.7645354,5.4819613,18.099218,2.2249596,4.5037537,2.696809,1.7692019,5.423552,7.2483773,1.7056755,2.245639,8.246046,0,8.395122,0,0,14.41051,16.23702,0,2.8700268,3.226648]
    # radarsat_ndbc = [13.53290363,8.570838966,2.180406444,18.83329089,5.451016109,7.522402231,5.451016109,3.815711276,3.597670632,4.46983321,1.199223544,2.50746741,5.839185738,9.698580935,7.35737403,11.61574228,10.8263229,10.8263229,8.119742178,8.683613163,5.560036431,1.635304833]
    #
    # wspd_mean_bias = mean_bias(radarsat_ndbc, radarsat)
    # wspd_abs_mean_bias = abs_mean_bias(radarsat_ndbc, radarsat)
    # wspd_rmse = rmse(radarsat_ndbc, radarsat)
    # wspd_co = co(radarsat_ndbc, radarsat)
    # wspd_content = 'Number of Matches: ' + str(len(radarsat)) + \
    #                '\nMAE (m/s): ' + str(round(wspd_abs_mean_bias, 3)) + \
    #                '\nMBE (m/s): ' + str(round(wspd_mean_bias, 3)) + \
    #                '\nRMSE (m/s): ' + str(round(wspd_rmse, 3)) + \
    #                '\nCC: ' + str(round(wspd_co, 3))
    # print("Radarsat:\n", wspd_content)
    # plot_scatter_wspd_sar(envisat_ndbc, envisat, content=wspd_content, name='sar/radarsat_wspd', sate='Radarsat')


