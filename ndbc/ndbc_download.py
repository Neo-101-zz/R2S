# !/usr/bin/env python

"""Download Continuous Wind data and buoy station information."""

import urllib
import urllib.request
import re
import requests
from bs4 import BeautifulSoup
import os
import time


url_base = 'http://www.ndbc.noaa.gov/data/historical/cwind/'
params_file = {'filename': '4104122017.txt.gz', 'dir': 'data/cwind/Feb/'}
params_station = {'station': '41040'}
station_page = 'http://www.ndbc.noaa.gov/station_page.php'


def Schedule(a,b,c):
    """Schedule parameters for download.

    Parameters
    ----------
    a : ?
        Data block that has been downloaded.
    b : ?
        Size of data block.
    c : ?
        Size of remote file.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    per = 100.0 * a * b / c
    if per > 100 :
        per = 100
    print('%.2f%%' % per)


def get_station(year):
    """Get stations' id in specified year.

    Parameters
    ----------
    year : str
        String of year represented by 4 digits.

    Returns
    -------
    station_list : list of str
        List of stations' id in specified year.

    """
    request = urllib.request.Request(url_base)
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    result = re.findall(re.compile(r'<a href="(.*?).txt.gz">'), content)
    station_list = []
    for r in result:
        id = r[0:5]
        date = r[6:10]
        if date == year:
            station_list.append(id)
    return station_list

def get_information(station, year):
    """Download station information.

    Parameters
    ----------
    station : str
        Station's id.
    year : str
        Specified year.

    Returns
    -------
    bool or str : 
        'error' if fail getting html.  True if information is got.  
        False if information is too less.

    Notes
    -----
    The parameter `year` may be unnecessary because it is not differed
    according to the year.

    """
    print('get_information'+station+year)
    filename = './information/'+year+'/'+station+'information.txt'
    if os.path.exists(filename):
        return True
    params_station['station'] = str.lower(station)
    try:
        html = requests.get(station_page, params=params_station, verify=False)
        print (html)
    except:
        return 'error'
    page = BeautifulSoup(html.text)
    div = page.find_all('div', id='stn_metadata')
    div = BeautifulSoup(str(div))
    information = div.find_all('p')
    if len(information) < 2:
        return False
    else:
        write_information(filename, information[1].text.replace('\xa0'*8, '\n\n'))
        print('information end: '+station+year)
        return True

def get_data(station, year):
    """Download Continuous Wind data of specified station and year.

    Parameters
    ----------
    station : str
        Station id.
    year : str
        Specified year.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    print('get_data'+station+year)
    local_path = './data/'+year+'/'+station+'c'+year+'.txt.gz'
    if os.path.exists(local_path):
        return True
    down_url = url_base + station+'c'+year+'.txt.gz'
    urllib.request.urlretrieve(down_url, local_path, Schedule)

def write_information(filename, data):
    """Write data by function open() with 'a' mode."""
    with open(filename, 'a') as f:
        f.write(data)

if __name__ == '__main__':
    year_list = ['2015', '2016', '2017']
    for year in year_list:
        station_list = get_station(year)
        for station in station_list:
            result = get_information(station, year)
            i = 1
            while result == 'error':
                print('reconnect: %d' % i)
                result = get_information(station, year)
                i += 1
        #     get_data(station, year)
        #     write_information('./'+year+'.txt', url_base + station + 'c'+year+'.txt.gz\n')
        #     print(year+station)
