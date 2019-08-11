#!/usr/bin/python
#encoding:utf-8

from datetime import date, timedelta
import os
import pickle

import numpy as np

import conf_satel
import dl_util

"""
Dates of operation
------------------
ASCAT : 2006 - present
QuickSCAT : 1999 - 2009
Windsat : 2003 - present

Range of dates
--------------
ASCAT : 2007/03/01 - present
QucikSCAT : 1999/07/19 - 2009/11/19
Windsat : 2003/02/05 - present

"""

def date_filter(input):
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

def update_missing_dates(missing_dates_file, missing_date):
    if os.path.exists(missing_dates_file):
        with open(missing_dates_file, 'rb') as fr:
            missing_dates = pickle.load(fr)
    else:
        missing_dates = []

    missing_dates.append(missing_date)
    with open(missing_dates_file, 'wb') as fw:
        pickle.dump(missing_dates, fw)

def get_data(sate_name, data_url, file_suffix, save_dir,
             start_date, end_date, missing_dates, missing_dates_file):
    os.makedirs(save_dir, exist_ok=True)
    delta_date = end_date - start_date
    for i in range(delta_date.days + 1):
        date_ = start_date + timedelta(days=i)
        # if date_ in missing_dates:
        #     continue
        file_name = '%s_%04d%02d%02d%s' % (sate_name, date_.year,
                                           date_.month, date_.day,
                                           file_suffix)
        file_url = '%sy%04d/m%02d/%s' % (data_url, date_.year,
                                         date_.month, file_name)
        if not dl_util.url_exists(file_url):
            # if not os.path.exists(missing_dates_file):
            print('Missing date: ' + str(date_))
            update_missing_dates(missing_dates_file, date_)
            continue
        file_path = save_dir + file_name
        dl_util.download(file_url, file_path)

def main():
    confs = conf_satel.configure()
    print(confs['choose_satellite_prompt'], end='')
    sate_confs = confs['spec_confs'][int(input()) - 1]
    start_date = date_filter(input(confs['input_date_prompt'][0]))
    end_date = date_filter(input(confs['input_date_prompt'][1]))
    dl_util.set_format_custom_text(confs['data_name_len'])
    print()
    get_data(sate_confs['name'], sate_confs['data_url'],
             sate_confs['file_suffix'], sate_confs['data_dir'],
             start_date, end_date, sate_confs['missing_dates'],
             sate_confs['missing_dates_file'])
    print()

if __name__ == '__main__':
    dl_util.arrange_signal()
    main()
