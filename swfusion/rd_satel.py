import os
import math
import pickle
from datetime import date
import calendar

import numpy as np
import matplotlib.pyplot as plt

import rd_util
import load_config

def monthly_batch_read_daily_satel(satel_name, year, month, data_dir,
                                   file_suffix, lats, lons):
    num_days = calendar.monthrange(year, month)[1]
    dates = [date(year, month, day) for day in range(1, num_days+1)]
    data = {}
    for idx, date_ in enumerate(dates):
        date_str = date_.strftime('%Y%m%d')
        file_path = '%s%s_%s%s' % (data_dir, satel_name, date_str,
                                    file_suffix)
        if not os.path.exists(file_path):
            data[date_.day] = []
            # print('File not exist: ' + file_path)
            continue
        dataset = read_util.read_daily_satel(satel_name, file_path)
        data[date_.day] = read_util.cut_map(
            satel_name, dataset, lats, lons, year, month, date_.day)

    return data

def check_read_date(CONFIG, satel_name, read_date):
    """Check whether monthly date of satellite data being read is in
    corresponding satellite's operation range.

    """
    start_limit = CONFIG['operation_dates'][satel_name]['start']
    # Expand start limit of date to the beginning of its month
    start_limit = start_limit.replace(day=1)
    end_limit = CONFIG['operation_dates'][satel_name]['end']
    if end_limit.year == 9999:
        end_limit = date.today()
    # Expand end limit of date to the end of its month
    try:
        last_day = calendar.monthrange(end_limit.year, end_limit.month)[1]
        end_limit = end_limit.replace(day=last_day)
    except ValueError:
        breakpoint()
        print(satel_name)
        print(end_limit)
        print(last_day)
        exit()

    if read_date < start_limit:
        print(('[Skip] For '
               + satel_name
               + read_date.strftime(', %Y/%m ')
               + CONFIG['prompt']['satel']['error']['date_too_early']
               + start_limit.strftime(' %Y/%m.')))
        return False
    if read_date > end_limit:
        print(('[Skip] For '
               + satel_name
               + read_date.strftime(', %Y/%m ')
               + CONFIG['prompt']['satel']['error']['date_too_late']
               + end_limit.strftime(' %Y/%m.')))
        return False

    return True

def empty_monthly_data(monthly_data):
    for day in monthly_data.keys():
        if monthly_data[day]:
            return False
    return True

def show_daily_data_existence(monthly_data):
    not_empty = []
    for day in monthly_data.keys():
        if monthly_data[day]:
            not_empty.append(day)
    # [1, 2, 3]
    show_str = []
    start = None
    end = None
    for idx, day in enumerate(not_empty):
        if not start:
            start = day
        if not idx:
            continue
        if not_empty[idx-1] == day - 1:
            end = day
        else:
            if end:
                show_str.append('%d~%d, ' % (start, end))
                start = None
                end = None
            else:
                show_str.append('%d, ' % start)
                start = None
    if end:
        show_str.append('%d~%d, ' % (start, end))
        start = None
        end = None
    else:
        show_str.append('%d, ' % start)
        start = None

    print((''.join(show_str))[:-2])

def yearly_dump_daily_satel(CONFIG, satel_name, year, lats, lons):
    print('\nRead %s\'s data\n' % satel_name)
    file_suffix = CONFIG['data_suffix'][satel_name]
    data_dir = CONFIG['dirs'][satel_name]['bmaps']
    pickle_dir = CONFIG['dirs'][satel_name]['pickle']
    os.makedirs(pickle_dir, exist_ok=True)

    for month in range(1, 13):
        read_date = date(year, month, 15)
        if not check_read_date(CONFIG, satel_name, read_date):
            continue
        monthly_data = monthly_batch_read_daily_satel(
            satel_name, year, month, data_dir,
            file_suffix, lats, lons)
        if empty_monthly_data(monthly_data):
            print(('[Skip] Empty monthly pickle file: '
                   + read_date.strftime('%Y/%m')))
            continue
        print(('[Hit] Read '
               + read_date.strftime('%Y/%m')
               + ' successfully. Valid days: '), end='')
        show_daily_data_existence(monthly_data)
        pickle_path = '%s%s_%d_%d.pkl' % (pickle_dir, satel_name,
                                          year, month)
        pickle_file = open(pickle_path, 'wb')
        pickle.dump(monthly_data, pickle_file)
        pickle_file.close()

def test(CONFIG):
    satel_name = ['ascat', 'qscat', 'wsat']
    year = [2007, 1999, 2003]
    month = [3, 7, 2]
    lats = (30, 50)
    lons = (220, 242)

    yearly_dump_daily_satel(CONFIG, satel_name[0], year[0], lats, lons)
    yearly_dump_daily_satel(CONFIG, satel_name[1], year[1], lats, lons)
    yearly_dump_daily_satel(CONFIG, satel_name[2], year[2], lats, lons)

def origin():
    import read_windsat as rw
    # Dump all windsat data into monthly pickle files belonging to
    # corresponding year directories
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
            # to skip file that is not gzip
            if file[0] not in ['1', '2']:
                continue
            month = int(file[4:6])
            day = int(file[6:8])
            dataset = rw.read_windsat(dir+'/'+file)
            if month == last_month:
                # continue add data into same month
                windsats_for_month += rw.cut_map(dataset, month, day)
            else:
                if last_month == 0:
                    # first iteration
                    windsats_for_month = []
                    windsats_for_month += rw.cut_map(dataset, month, day)
                    last_month = month
                else:
                    # save last month's data into a pickle file
                    pickle_file = open(cur_dir + '/windsat_'
                                       + str(last_month) + '.pkl', 'wb')
                    pickle.dump(windsats_for_month, pickle_file)
                    pickle_file.close()
                    # read new month's data
                    windsats_for_month = []
                    windsats_for_month += rw.cut_map(dataset, month, day)
                    last_month = month
        pickle_file = open(cur_dir + '/windsat_' + str(last_month) + '.pkl', 'wb')
        pickle.dump(windsats_for_month, pickle_file)
        pickle_file.close()

if __name__ == '__main__':
    CONFIG = load_config.load_config()
    test(CONFIG)
