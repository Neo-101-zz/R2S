import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

import load_configs
import utils

TIME_D = 1440
LON_D = 1440
LAT_D = 720
ORI_D = 2
TIME_GAP = 1
LAT_GAP = 0.25
LON_GAP = 0.25

def test_mingmt_duplicate_and_its_vars(vars):
    """Test whether there are duplicated mingmt in 2 orbits of
    a daily bytemap file or not and if corrsponding variables
    in 2 oribits is same.

    """
    same_count = 0
    vars_same = True
    same_mingmt = []
    same_lat = []
    same_lon = []
    for j in range(LAT_D):
        for k in range(LON_D):
            if vars['nodata'][0][j][k] or vars['nodata'][1][j][k]:
                continue
            if vars['mingmt'][0][j][k] == vars['mingmt'][1][j][k]:
                breakpoint()
                same_count += 1
                same_mingmt.append(int(vars['mingmt'][0][j][k]))
                same_lat.append(float(vars['latitude'][j]))
                same_lon.append(float(vars['longitude'][k]))
                if vars_same:
                    for var in vars.keys():
                        if var == 'latitude' or var == 'longitude':
                            continue
                        if vars[var][0][j][k] != vars[var][1][j][k]:
                            vars_same = False
    print('[Result] Same mingmt pairs count: %d' % same_count)
    if same_count:
        print('[Result] Variables of same mingmt pairs are same: %s' % vars_same)

    same_mingmt = np.array(same_mingmt)
    same_lat = np.array(same_lat)
    same_lon = np.array(same_lon)

    return same_mingmt, same_lat, same_lon

def plot_var_distribution(save_dir, satel_name, date_, vars):
    mingmt = np.array([])
    for i in range(ORI_D):
        for j in range(LAT_D):
            for k in range(LON_D):
                pass

def plot_same_distribution(save_dir, satel_name, date_, same_mingmt,
                           same_lat, same_lon):
    os.makedirs(save_dir, exist_ok=True)

    bin = 2 * int((np.amax(same_mingmt) - np.amin(same_mingmt)) / TIME_GAP) + 1
    bin = 20 if bin < 20 else bin
    plt.hist(same_mingmt, bin, facecolor='g', alpha=0.75)
    plt.xlabel('Mingmt')
    plt.ylabel('Count')
    plt.title('Histogram of same mingmt of {0} in {1}'.format(
              satel_name.upper(), date_.strftime('%Y/%m/%d')))
    plt.grid(True)
    plt.savefig('{0}hist_of_same_mingmt_{1}_{2}.png'.format(
        save_dir, satel_name, date_.strftime('%Y_%m_%d')))
    plt.clf()

    bin = 2 * int((np.amax(same_lat) - np.amin(same_lat)) / LAT_GAP) + 1
    breakpoint()
    bin = 20 if bin < 20 else bin
    plt.hist(same_lat, bin, facecolor='g', alpha=0.75)
    plt.xlabel('Latitude')
    plt.ylabel('Count')
    plt.title(('Histogram of same latitude at same mingmt of {0} in '
               + '{1}').format(satel_name.upper(), date_.strftime('%Y/%m/%d')))
    plt.grid(True)
    plt.savefig('{0}hist_of_same_latitude_at_same_mingmt_{1}_{2}.png'.format(
        save_dir, satel_name, date_.strftime('%Y_%m_%d')))
    plt.clf()

    bin = 2 * int((np.amax(same_lon) - np.amin(same_lon)) / LON_GAP) + 1
    bin = 20 if bin < 20 else bin
    plt.hist(same_lon, bin, facecolor='g', alpha=0.75)
    plt.xlabel('Longitude')
    plt.ylabel('Count')
    plt.title(('Histogram of same longitude at same mingmt of {0} in '
               + '{1}').format(satel_name.upper(), date_.strftime('%Y/%m/%d')))
    plt.grid(True)
    plt.savefig('{0}hist_of_same_longitude_at_same_mingmt_{1}_{2}.png'.format(
        save_dir, satel_name, date_.strftime('%Y_%m_%d')))
    plt.clf()

def count_num_of_one_mingmt(mingmt):
    """Count how many times specified mingmt occurs in a dataset.

    """
    time = int(input('[Input] Enter the mingmt you want to count: '))
    time_count = 0
    for i in range(ORI_D):
        for j in range(LAT_D):
            for k in range(LON_D):
                if int(mingmt[i][j][k]) == time:
                    time_count += 1
    print('[Result] Count of mingmt %d: %d' % (time, time_count))

def load_data(CONFIG, satel_name, date):
    """Load satellite bytemap dataset.

    """
    dir = CONFIG[satel_name]['dirs']['bmaps']
    suffix = CONFIG[satel_name]['data_suffix']
    date_str = date.strftime('%Y%m%d')
    file_path = '{0}{1}_{2}{3}'.format(dir, satel_name, date_str, suffix)
    dataset = utils.dataset_of_daily_satel(satel_name, file_path)
    print('[Info] [%s] Load %s' % (satel_name.upper(), file_path))

    return dataset

def input_args():
    satel = input('[Input] Enter satellite name: ')

    date_ = utils.filter_datetime(
        input('[Input] Enter date in form of year/month/day: ') + '/0/0/0').date()

    return satel, date_

def choose_mode():
    print(('1. Custom mode\n'
           + '2. Entirely check mode\n'))
    choice = int(input('Enter your choice: '))
    return choice

def custom_mode(CONFIG):
    satel_name, date_ = input_args()
    dataset = load_data(CONFIG, satel_name, date_)
    count_num_of_one_mingmt(dataset.variables['mingmt'])
    test_mingmt_duplicate_and_its_vars(dataset.variables)

def entirely_check_mode(CONFIG):
    for satel_name in ['ascat', 'qscat', 'wsat']:
        dir = CONFIG[satel_name]['dirs']['bmaps']
        files = [x for x in os.listdir(dir) if x.endswith('.gz')]
        dates_str = [x.split('_')[1][:8] for x in files]
        dates = [datetime.datetime.strptime(x+'000000',
                                           '%Y%m%d%H%M%S').date() \
                 for x in dates_str]
        for date_ in dates:
            dataset = load_data(CONFIG, satel_name, date_)
            same_mingmt, same_lat, same_lon = \
                    test_mingmt_duplicate_and_its_vars(dataset.variables)
            if len(same_mingmt):
                plot_same_distribution(CONFIG['result']['dirs']['fig'],
                                       satel_name, date_, same_mingmt,
                                       same_lat, same_lon)

if __name__ == '__main__':
    CONFIG = load_configs.load_config()
    mode = choose_mode()
    if mode == 1:
        custom_mode(CONFIG)
    elif mode == 2:
        entirely_check_mode(CONFIG)
    # test(time)
