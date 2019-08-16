# !/usr/bin/env python

"""Download NDBC Continuous Wind data and station information.

"""
import re
import requests
import os
import time
import pickle

from bs4 import BeautifulSoup

import load_config
import dl_util

def station_in_a_year(year, url, candidate_stations=None):
    """Get stations' id in specified year.

    Parameters
    ----------
    year : str
        String of year represented by 4 digits.
    url : str
        Url of Continuous Wind data directory.
    candidate_stations : set, optional
        Stations which possibly emerged in the `year`.

    Returns
    -------
    stations : set of str
        Set of stations' id in a specified year.

    """
    page = requests.get(url)
    data = page.text
    soup = BeautifulSoup(data, features='lxml')
    stations = set()
    suffix = 'c%s.txt.gz' % year
    anchors = soup.find_all('a')

    if not candidate_stations:
        for link in anchors:
            href = link.get('href')
            if href.endswith(suffix):
                id = href[0:5]
                stations.add(id)
    else:
        for link in anchors:
            href = link.get('href')
            id = href[0:5]
            if href.endswith(suffix) and id in candidate_stations:
                stations.add(id)

    return stations

def year_of_a_station(station, url):
    """Get years when the data of specified station existed.

    Parameters
    ----------
    station : str
        String of station id.
    url : str
        Url of Continuous Wind data directory.

    Returns
    -------
    years : set of str
        Set of years of a specified station.

    """
    page = requests.get(url)
    data = page.text
    soup = BeautifulSoup(data, features='lxml')
    years = set()
    prefix = station
    suffix = '.txt.gz'
    anchors = soup.find_all('a')

    for link in anchors:
        href = link.get('href')
        if href.startswith(prefix) and href.endswith(suffix):
            year = href[6:10]
            years.add(year)

    return years

def dl_station_info(station, url, save_dir):
    """Download station information.

    Parameters
    ----------
    station : str
        Station's id.
    save_dir : str
        Directory of files saving station information.

    Returns
    -------
    bool or str : 
        'error' if fail getting html.  True if information is got.  
        False if information is too less.

    """
    file_path = save_dir + station + '.txt'
    if os.path.exists(file_path):
        return

    payload = dict()
    payload['station'] = str.lower(station)
    try:
        html = requests.get(url, params=payload, verify=True)
    except:
        return 'error'
    page = BeautifulSoup(html.text, features='lxml')
    div = page.find_all('div', id='stn_metadata')
    div = BeautifulSoup(str(div), features='lxml')
    information = div.find_all('p')
    if len(information) < 2:
        return False
    else:
        # write_information(file_path, information[1].text.replace('\xa0'*8, '\n\n'))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(information[1].text)
        print(station)
        return True

def dl_cwind_data(station, year, save_dir, data_url):
    """Download Continuous Wind data of specified station and year.

    Parameters
    ----------
    station : str
        Station id.
    year : str
        Specified year.
    save_dir : str
        Directory to save CWind data.
    data_url : str
        Url of data folder which stores data files.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    os.makedirs(save_dir, exist_ok=True)
    file_name = '{0}c{1}.txt.gz'.format(station, year)
    file_path = '{0}{1}'.format(save_dir, file_name)
    file_url = '{0}{1}'.format(data_url, file_name)
    dl_util.download(file_url, file_path)

def save_relation(path, var):
    """Append variable to pickle file.
    
    Parameters
    ----------
    path : str
        Location of pickle file to store the relation.
    var : dict
        Store the relation in the form of dict, of which key is str and
        value is set of str.
    
    Returns
    -------
    None
        Nothing returned by this function.
    
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # read relation if it exists
    if os.path.exists(path):
        with open(path, 'rb') as fr:
            rel = pickle.load(fr)
        # append var to existing relation
        for key in var.keys():
            if key in rel:
                rel[key].update(var[key])
            else:
                rel[key] = var[key]
    else:
        rel = var

    with open(path, 'wb') as fw:
        pickle.dump(rel, fw)

def filter_year(input):
    """Filter the inputted year.
    
    Parameters
    ----------
    input : str
        Inputted string of target year.
    
    Returns
    -------
    set of int
        Return a set of str, e.g. {1997, 1999, 2000}.
    
    """
    if not input:
        return set()
    if not input.count(' '):
        # single input
        if not input.count('-'):
            res = set()
            res.add(int(input))
        # range input
        else:
            begin = int(input.split('-')[0])
            end = int(input.split('-')[1])
            res = set([x for x in range(begin, end+1)])
    # hybrid input
    else:
        parts = input.replace('  ', ' ').split(' ')
        res = set()
        for part in parts:
            temp = filter_year(part)
            res.update(temp)

    return res

def filter_station(input):
    """Filter the inputted station id.
    
    Parameters
    ----------
    input : str
        Inputted string of target station id.
    
    Returns
    -------
    set of str
        Return a set of str, e.g. {'41001', '41002', 'lonf1'}.
    
    """
    if not input:
        return set()
    # single input
    if not input.count(' '):
        res = set()
        res.add(input)
    # multi input
    else:
        res = set(input.replace('  ', ' ').split(' '))

    return res

def input_year(CONFIG):
    print(CONFIG['prompt']['ndbc']['input']['year'], end='')
    years = filter_year(input())

    return years

def input_station(CONFIG):
    print(CONFIG['prompt']['ndbc']['input']['station'], end='')
    stations = filter_station(input())

    return stations

def analysis_and_save_relation(CONFIG, years, stations):
    """Analysis and save relation between inputted year(s) and station(s)
    from NDBC's Continuous Wind Historical Data webpage.

    """
    # key: station, value: year
    station_year = dict()
    # key: year, value: station
    year_station = dict()
    # year is specified but station is not
    if years and not stations:
        # Collect stations' id according to years
        for year in years:
            stns = station_in_a_year(year, CONFIG['urls']['ndbc']['cwind'])
            year_station[year] = stns
            stations.update(stns)
            for stn in stns:
                if not stn in station_year:
                    station_year[stn] = set()
                station_year[stn].add(year)
    # station is specified but year is not
    elif not years and stations:
        # Collect years according to stations' id
        for stn in stations:
            yrs = year_of_a_station(stn, CONFIG['urls']['ndbc']['cwind'])
            station_year[stn] = yrs
            years.update(yrs)
            for yr in yrs:
                if not yr in year_station:
                    year_station[yr] = set()
                year_station[yr].add(stn)
    # both year and station is specified
    elif years and stations:
        # No need to update years and stations
        for year in years:
            stns = station_in_a_year(year, CONFIG['urls']['ndbc']['cwind'],
                                     candidate_stations=stations)
            year_station[year] = stns
            for stn in stations:
                if not stn in station_year:
                    station_year[stn] = set()
                station_year[stn].add(year)
    else:
        exit(CONFIG['prompt']['ndbc']['error']['no_input'])
    # Save two dicts which store the relation between stations and year
    save_relation(CONFIG['vars_path']['ndbc']['year_station'],
                  year_station)
    save_relation(CONFIG['vars_path']['ndbc']['station_year'],
                  station_year)

    return year_station

def download_station_info(CONFIG, stations):
    """Download all stations' information into single directory.

    """
    print(CONFIG['prompt']['ndbc']['info']['dl_station'])
    for stn in stations:
        i = 0
        while True:
            # download stations' information
            result = dl_station_info(
                stn, CONFIG['urls']['ndbc']['stations'],
                CONFIG['dirs']['ndbc']['stations'])
            if result != 'error':
                break
            else:
                print('Fail downloading station'
                  + ' {0}\'s information.'.format(stn))
                i += 1
                if i <= CONFIG['retry_times']['ndbc']: 
                    print('reconnect: %d' % i)
                else:
                    exit(CONFIG['error']['no_station_info'])

def download_cwind_data(CONFIG, years, year_station):
    """Download Continuous Wind data into single directory.

    """
    print(CONFIG['prompt']['ndbc']['info']['dl_cwind'])
    dl_util.set_format_custom_text(CONFIG['data_name_length']['ndbc'])
    for year in years:
        for stn in year_station[year]:
            dl_cwind_data(stn, year, CONFIG['dirs']['ndbc']['cwind'],
                          CONFIG['urls']['ndbc']['cwind'])

    print('\n')

def download_ndbc(CONFIG):
    """Download NDBC's buoys station's information and corresponding
    Continuous Wind Historical Data according to user's input of
    year(s) and station(s).

    """
    # Set of int
    years = input_year(CONFIG)
    # Set of str
    stations = input_station(CONFIG)
    year_station = analysis_and_save_relation(CONFIG, years, stations)
    # Due to update of set in function analysis_and_save_relation,
    # variable stations is the set of all stations corresponding to
    # variable years now
    download_station_info(CONFIG, stations)
    download_cwind_data(CONFIG, years, year_station)

if __name__ == '__main__':
    CONFIG = load_config.load_config()
    dl_util.arrange_signal()
    download_ndbc(CONFIG)
