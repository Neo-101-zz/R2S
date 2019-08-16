# !/usr/bin/env python
"""Download SFMR data of hurricane from NOAA HRD.

"""
import re
import requests
import os
import pickle
from datetime import date

from bs4 import BeautifulSoup

import load_config
import dl_util
import dl_ndbc

def gen_year_hurr(CONFIG, years):
    year_hurr = {}

    for year in years:
        if int(year) < 1994:
            year = 'prior1994'
        url = '{0}{1}.html'.format(
            CONFIG['urls']['sfmr']['hurricane'][:-5], year)
        page = requests.get(url)
        data = page.text
        soup = BeautifulSoup(data, features='lxml')
        anchors = soup.find_all('a')

        year_hurr[year] = set()

        for link in anchors:
            if not link.contents:
                continue
            text = link.contents[0]
            if text != 'SFMR':
                continue
            href = link.get('href')
            hurr = href.split('/')[-2][:-4]
            year_hurr[year].add(hurr)

    return year_hurr

def filter_year_hurr_input(input):
    """Filter input to get a dict of hurricane name and year.

    Parameters
    ----------
    input : str
        Inputted string.

    Returns
    -------
    year_hurr : dict
        A dict with year as key and set of corresponding hurricane name
        as value.

    """
    year_hurr = {}
    input = input.replace(' ', '').lower()
    parts = re.findall('\[(.*?)\]', input)
    for part in parts:
        if part.count(','):
            year = part.split(',')[0]
            hurr = part.split(',')[1]
            if not year in year_hurr:
                year_hurr[year] = set()
            year_hurr[year].add(hurr)
        else:
            exit('Input format error.')

    return year_hurr

def download_sfmr_data(CONFIG, year_hurr, period=None):
    """Download SFMR data of hurricanes.

    Parameters
    ----------
    year_hurr : dict
        A dict with year as key and set of corresponding hurricane name
        as value.
    CONFIG : dict
        A dict of configuration.

    Returns
    -------
    hit_times : dict
        Times of hurricane NetCDF file's date being in period.

    """
    print(CONFIG['prompt']['sfmr']['info']['dl_hurr'])
    dl_util.set_format_custom_text(CONFIG['data_name_length']['sfmr'])
    suffix = '.nc'
    save_root_dir = CONFIG['dirs']['sfmr']['hurr']
    os.makedirs(save_root_dir, exist_ok=True)

    hit_times = {}
    for year in year_hurr.keys():
        hit_times[year] = {}
        # Create directory to store SFMR files
        os.makedirs('{0}{1}'.format(save_root_dir, year), exist_ok=True)
        hurrs = list(year_hurr[year])
        for hurr in hurrs:
            dir_path = '{0}{1}/{2}/'.format(
                save_root_dir, year, hurr)
            os.makedirs(dir_path, exist_ok=True)
            # Generate keyword to consist url
            keyword = '{0}{1}'.format(hurr, year)
            url = '{0}{1}{2}'.format(CONFIG['urls']['sfmr']['prefix'], keyword,
                                     CONFIG['urls']['sfmr']['suffix'])
            # Get page according to url
            page = requests.get(url)
            data = page.text
            soup = BeautifulSoup(data, features='lxml')
            anchors = soup.find_all('a')

            # Times of NetCDF file's date being in period
            hit_count = 0
            for link in anchors:
                href = link.get('href')
                # Find href of netcdf file
                if href.endswith(suffix):
                    # Extract file name
                    file_name = href.split('/')[-1]
                    date_str = file_name[-13:-5]
                    date_ = date(int(date_str[:4]),
                                 int(date_str[4:6]),
                                 int(date_str[6:]))
                    if not dl_util.check_period(date_, period):
                        continue
                    hit_count += 1
                    file_path = dir_path + file_name
                    dl_util.download(href, file_path)
            hit_times[year][hurr] = hit_count
    
    return hit_times

def save_year_hurr(CONFIG, year_hurr, hit_times, strict_mode=False):
    """Append year_hurr to the pickle file which store this variable.
    
    Parameters
    ----------
    path : str
        Location of pickle file to store the variable.
    year_hurr : dict
        A dict with year as key and set of corresponding hurricane name
        as value.
    
    Returns
    -------
    None
        Nothing returned by this function.
    
    """
    if strict_mode:
        for year in hit_times.keys():
            for hurr in hit_times[year].keys():
                if not hit_times[year][hurr]:
                    year_hurr[year].remove(hurr)
                    if not year_hurr[year]:
                        year_hurr.pop(year)
                        if not year_hurr:
                            return

    path = CONFIG['vars_path']['sfmr']['year_hurr']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # read relation if it exists
    if os.path.exists(path):
        with open(path, 'rb') as fr:
            old_year_hurr = pickle.load(fr)
        for year in year_hurr.keys():
            if year in old_year_hurr:
                year_hurr[year].update(old_year_hurr[year])

    new_year_hurr = year_hurr

    with open(path, 'wb') as fw:
        pickle.dump(new_year_hurr, fw)

def download_sfmr(CONFIG):
    # print(CONFIG['prompt']['sfmr']['input']['hurr'], end='')
    # year_hurr = filter_year_hurr_input(input())
    years = dl_ndbc.input_year(CONFIG)
    year_hurr = gen_year_hurr(CONFIG, years)

    period = [date(2002,1,1), date(2002,9,25)]
    hit_times = download_sfmr_data(CONFIG, year_hurr, period)
    save_year_hurr(CONFIG, year_hurr, hit_times)
    print()

if __name__ == '__main__':
    CONFIG = load_config.load_config()
    dl_util.arrange_signal()
    # Original author's target hurricane SFMR data:
    # [2011, Dora] [2014, Simon] [2015, Guillermo]
    # [2015, Patricia] [2016, Javier]
    download_sfmr(CONFIG)
