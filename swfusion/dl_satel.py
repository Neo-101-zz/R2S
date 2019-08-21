#!/usr/bin/python

from datetime import date, timedelta
import os
import pickle

import numpy as np

import load_config
import dl_util

def filter_date(input):
    """Filter the inputted date.

    Parameters
    ----------
    input : str
    Inputted string of date in the form of YEAR/MONTH/DAY.

    Returns
    -------
    date
    An idealized naive date in the form of current Gregorian
    calendar.

    """
    year, month, day = input.split('/')
    while month.startswith('0'):
        month = month[1:]
        while day.startswith('0'):
            day = day[1:]

    return date(int(year), int(month), int(day))

def check_satel_period(CONFIG, satel_name, period):
    """Check whether inputted start date and end date is in corresponding
    satellite's operation range.

    """
    start_date = period[0]
    end_date = period[1]
    start_limit = CONFIG['operation_dates'][satel_name]['start']
    end_limit = CONFIG['operation_dates'][satel_name]['end']
    if end_limit.year == 9999:
        end_limit = date.today()
    if start_date < start_limit:
        print((CONFIG['prompt']['satel']['error']['range_too_early']
               + start_limit.strftime(' %Y/%m/%d.\n')))
        return False
    if end_date > end_limit:
        print((CONFIG['prompt']['satel']['error']['range_too_late']
               + end_limit.strftime(' %Y/%m/%d.\n')))
        return False
    if start_date > end_date:
        print('[Error] Start date is later than end date')
        return False
    return True

def download_satel_data(CONFIG, satel_name, period):
    """Download ASCAT/QucikSCAT/Windsat data in specified date range.

    """
    start_date = period[0]
    end_date = period[1]
    data_url = CONFIG['urls'][satel_name]
    file_suffix = CONFIG['data_suffix'][satel_name]
    save_dir = CONFIG['dirs'][satel_name]['bmaps']
    missing_dates_file = CONFIG['files_path'][satel_name]['missing_dates']

    print('\nDownload %s data\n' % satel_name)
    dl_util.set_format_custom_text(CONFIG['data_name_length']['satel'])
    if os.path.exists(missing_dates_file):
        with open(missing_dates_file, 'rb') as fr:
            missing_dates = pickle.load(fr)
    else:
        missing_dates = set()

    os.makedirs(save_dir, exist_ok=True)
    delta_date = end_date - start_date
    for i in range(delta_date.days + 1):
        date_ = start_date + timedelta(days=i)
        if date_ in missing_dates:
            continue
        file_name = '%s_%04d%02d%02d%s' % (
            satel_name, date_.year, date_.month, date_.day, file_suffix)
        file_url = '%sy%04d/m%02d/%s' % (
            data_url, date_.year, date_.month, file_name)
        if not dl_util.url_exists(file_url):
            print('Missing date: ' + str(date_))
            print(file_url)
            missing_dates.add(date_)
            with open(missing_dates_file, 'wb') as fw:
                pickle.dump(missing_dates, fw)
            continue

        file_path = save_dir + file_name
        dl_util.download(file_url, file_path)
    print()

def choose_satel(CONFIG):
    satel_names = ['ascat', 'qscat', 'wsat']
    print(CONFIG['prompt']['satel']['info']['choose_satel'])
    print(CONFIG['prompt']['satel']['input']['satel_num'], end='')
    satel_name = satel_names[int(input()) - 1]

    return satel_name

def input_period(CONFIG, satel_name=None, check=True):
    print(CONFIG['prompt']['satel']['info']['period'])
    start_date = filter_date(
        input(CONFIG['prompt']['satel']['input']['start_date']))
    end_date = filter_date(
        input(CONFIG['prompt']['satel']['input']['end_date']))
    if satel_name and check:
        if not check_satel_period(CONFIG, satel_name,
                                  [start_date, end_date]):
            exit()

    return start_date, end_date

def download_satel(CONFIG):
    """Receive user's choice between ASCAT/QucikSCAT/Windsat and set of
    date range to download corresponding data.

    """
    # Specify satellite's name with user's choice
    satel_name = choose_satel(CONFIG)
    start_date, end_date = input_period(CONFIG, satel_name)
    # Use satellite's name to index configuration
    download_satel_data(CONFIG, satel_name, [start_date, end_date])

if __name__ == '__main__':
    CONFIG = load_config.load_config()
    dl_util.arrange_signal()
    download_satel(CONFIG)
