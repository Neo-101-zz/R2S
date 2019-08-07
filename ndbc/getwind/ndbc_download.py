# !/usr/bin/env python

"""Download NDBC Continuous Wind data and station information.

Features
--------
input years :
    Support single input ('1997'), range input ('1997-2001'), and hybrid
    input ('1997 1999-2001 2003-2010').

input stations :
    Support single input ('41001'), multi-input ('41001, burl1').

select stations according to years :
    Do following steps in each run of program.  
    First derive stations' id from NDBC historical Continuous Wind data
    directory according to the inputted year.  
    Second store the relation between year and station in the form of 
    a dict with year as key and station's id as value.
    Third merge the dict into a pickle file which saves the relation
    in the same form of the dict for other program calls.  

select years according to stations :
    Do following steps in each run of program.
    First derive data's year from NDBC historical Continuous Wind data 
    directory according to the inputted station's id.  
    Second store the relation between station and year in the form of 
    a dict with year as key and station's id as value for other program
    calls.  
    Third merge the dict into a pickle file which saves the relation
    in the same form of the dict for other program calls.  

download station information :
    Extract text description of NDBC stations from NDBC website and save
    it as text file.

download cwind data :
    Save NDBC Continous Wind data according to the inputted year and
    station's id as txt.gz files.
"""

import urllib.request
import re
import requests
import os
import time
import pickle
import signal
import sys

from bs4 import BeautifulSoup
import progressbar

import ndbc_conf

# Global variables
pbar = None
format_custom_text = None
current_download_file = None

def handler(signum, frame):
    """Handle forcing quit which may be made by pressing Control + C and
    sending SIGINT which will interupt this application.

    Parameters
    ----------
    signum : int
        Signal number which is sent to this application.
    frame : ?
        Current stack frame.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    # Remove file that is downloaded currently in case forcing quit
    # makes this file uncomplete
    os.remove(current_download_file)
    # Print log
    print('\nForce quit on %s.\n' % signum)
    # Force quit
    sys.exit(1)

def set_format_custom_text(len):
    """Customize format text's length.

    Parameters
    ----------
    len : int
        Length of format text.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    global format_custom_text
    format_custom_text = progressbar.FormatCustomText(
        '%(f)-' + str(len) +'s ',
        dict(
            f='',
        ),
    )

def sizeof_fmt(num, suffix='B'):
    """Convert size of file from B to unit which let size value
    less than 1024.

    Parameters
    ----------
    num : float
        File size in bit.
    suffix : str, optional
        Character(s) after value of file size after convertion.  
        Default value is 'B'.

    Returns
    -------
    str
        File size after convertion.

    """
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Y', suffix)

def show_progress(block_num, block_size, total_size):
    """Show progress of downloading data with progress bar.

    Parameters
    ----------
    block_num : int
        Data block that has been downloaded.
    block_size : int
        Size of data block.
    total_size : int
        Size of remote file.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    global pbar
    global format_custom_text

    if pbar is None:
        pbar = progressbar.bar.ProgressBar(
            maxval=total_size,
            widgets=[
                format_custom_text,
                '  | %-8s  ' % sizeof_fmt(total_size),
                progressbar.Bar(marker=u'\u2588', fill='.',
                                left='| ', right= ' |'),
                progressbar.Percentage(),
            ])

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def get_station_in_a_year(year, url, candidate_stations=None):
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


def get_year_of_a_station(station, url):
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

def get_information(station, url, save_dir):
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
    filename = save_dir + station + '.txt'
    if os.path.exists(filename):
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
        # write_information(filename, information[1].text.replace('\xa0'*8, '\n\n'))
        write_information(filename, information[1].text)
        print(station)
        return True

def get_data(station, year, save_dir, url):
    """Download Continuous Wind data of specified station and year.

    Parameters
    ----------
    station : str
        Station id.
    year : str
        Specified year.
    save_dir : str
        Directory to save CWind data.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    os.makedirs(save_dir, exist_ok=True)
    file_name = '{0}c{1}.txt.gz'.format(station, year)
    local_path = '{0}{1}'.format(save_dir, file_name)
    global current_download_file
    current_download_file = local_path
    if os.path.exists(local_path):
        return True
    down_url = '{0}{1}'.format(url, file_name)
    
    global format_custom_text
    format_custom_text.update_mapping(f=file_name)
    urllib.request.urlretrieve(down_url, local_path, show_progress)

def write_information(path, data):
    """Write data by function open() with 'w' mode."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(data)

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

def year_filter(input):
    """Filter the inputted year.
    
    Parameters
    ----------
    input : str
        Inputted string of target year.
    
    Returns
    -------
    set of str
        Return a set of str, e.g. {'1997', '1999', '2000'}.
    
    """
    if not input:
        return set()
    if not input.count(' '):
        # single input
        if not input.count('-'):
            res = set()
            res.add(input)
        # range input
        else:
            begin = int(input.split('-')[0])
            end = int(input.split('-')[1])
            res = set([str(x) for x in range(begin, end+1)])
    # hybrid input
    else:
        parts = input.replace('  ', ' ').split(' ')
        res = set()
        for part in parts:
            temp = year_filter(part)
            res.update(temp)

    return res

def station_filter(input):
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

def main():
    confs = ndbc_conf.configure()
    print(confs['input_year_prompt'], end='')
    years = year_filter(input())
    print(confs['input_station_prompt'], end='')
    stations = station_filter(input())
    # key: station, value: year
    station_year = dict()
    # key: year, value: station
    year_station = dict()
    # year is specified but station is not
    if years and not stations:
        # Collect stations' id according to years
        for year in years:
            stns = get_station_in_a_year(year, confs['url_base'])
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
            yrs = get_year_of_a_station(stn, confs['url_base'])
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
            stns = get_station_in_a_year(year, confs['url_base'],
                                         candidate_stations=stations)
            year_station[year] = stns
            for stn in stations:
                if not stn in station_year:
                    station_year[stn] = set()
                station_year[stn].add(year)
    else:
        print('\nAt least one of year(s) and station(s) '
              + 'should be specified.\n')
        return
    # Save two dicts which store the relation between stations and year
    save_relation(confs['var_dir'] + 'year_station.pkl', year_station)
    save_relation(confs['var_dir'] + 'station_year.pkl', station_year)
    # Download all stations' information into single directory
    print('\nDownloading Station Information')
    for stn in stations:
        result = get_information(stn, confs['station_page'],
                                 confs['station_dir'])
        i = 1
        while result == 'error' and i <= confs['retry_times']:
            print('reconnect: %d' % i)
            result = get_information(stn, confs['station_page'],
                                     confs['station_dir'])
            i += 1
        if result == 'error':
            print('Fail downloading station'
                  +' {0}\'s information.'.format(stn))
    # Download Continuous Wind data
    print('\nDownloading Continuous Wind Data')
    set_format_custom_text(confs['data_name_len'])
    for year in years:
        for stn in year_station[year]:
            get_data(stn, year, confs['cwind_dir'], confs['url_base'])

    print('\n')

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGHUP, handler)
    signal.signal(signal.SIGTERM, handler)
    main()
