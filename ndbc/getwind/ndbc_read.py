# !/usr/bin/env python

import os
import gzip
import pickle
import math
import csv

import pandas as pd
import numpy as np

import ndbc_conf

station = {}
exception = ['41036information.txt', 'lscm4information.txt']

def extract_stn_info(gzip_file_name):
    """Read latitude, longitude and anemometer height of station from
    text file.

    Parameters
    ----------
    gzip_file_name : str
        Path of station text file.

    Returns
    -------
    station : dict
        Records station's latitude, longitude and anemometer height.

    """
    station = {}
    with open(gzip_file_name, 'r') as sf:
        # Assume the station information is not valid for using
        station['valid'] = False
        location = False
        site_elevation = False
        anemometer_height = False

        line_list = sf.readlines()

    for i in range(len(line_list)):
        line = line_list[i]

        if '°' in line and '\'' in line and '"' in line:
            location = True
            # Read latitude and longitude
            latlon_line = line
            is_north = (latlon_line[7:8] == 'N')
            is_east = (latlon_line[16:17] == 'E')
            if is_north:
                station['lat'] = float(latlon_line[0:6])
            else:
                station['lat'] = -float(latlon_line[0:6])
            if is_east:
                station['lon'] = float(latlon_line[9:15])
            else:
                station['lon'] = 360. - float(latlon_line[9:15])

        if 'Site elevation' in line:
            site_elevation = True
            # Read elevation
            site_line = line
            if site_line[16:] == 'sea level\n':
                site_height = 0
            else:
                strs = site_line[16:].split(' ')
                site_height = float(strs[0])

        if 'Anemometer height' in line:
            anemometer_height = True
            ane_line = line
            strs = ane_line[19:].split(' ')
            wind_height = site_height + float(strs[0])
            station['height'] = wind_height

    if location and site_elevation and anemometer_height:
        station['valid'] = True

    return station

def convert_10(wspd, height):
    """Convert the wind speed at the the height of anemometer to
    the wind speed at the height of 10 meters.

    Parameters
    ----------
    wspd : float
        Wind speed at the height of anemometer.
    height : float
        The height of anemometer.

    Returns
    -------
    con_wspd : float
        Wind speed at the height of 10 meters.

    References
    ----------
    Xiaoping Xie, Jiansu Wei, and Liang Huang, Evaluation of ASCAT
    Coastal Wind Product Using Nearshore Buoy Data, Journal of Applied
    Meteorological Science 25 (2014), no. 4, 445–453.

    """
    if wspd <= 7:
        z0 = 0.0023
    else:
        z0 = 0.022
    kz = math.log(10/z0) / math.log(height/z0)
    con_wspd = wspd * kz

    return con_wspd

if __name__ == '__main__':
    # Original author's setting
    # min_lat = 24 
    # max_lat = 52
    # min_lon = 230
    # max_lon = 240

    confs = ndbc_conf.configure()

    default_range = [confs['default_min_latitude'],
                     confs['default_max_latitude'],
                     confs['default_min_longitude'],
                     confs['default_max_longitude']]

    input_range = [input(confs['input_min_latitude_prompt']),
                   input(confs['input_max_latitude_prompt']),
                   input(confs['input_min_longitude_prompt']),
                   input(confs['input_max_longitude_prompt'])]

    for index, value in enumerate(input_range):
        if len(value) == 0:
            input_range[index] = default_range[index]
    min_lat, max_lat = input_range[0], input_range[1]
    min_lon, max_lon = input_range[2], input_range[3]
    # Collect all NDBC stations' info into a single csv file
    stations_csv_name = confs['station_csv']
    station_files = os.listdir(confs['station_dir'])
    stations_csv = open(stations_csv_name, 'w', newline='')
    writer = csv.writer(stations_csv)
    content = ['id', 'lat', 'lon', 'height']
    writer.writerow(content)

    for file in station_files:
        if '.txt' in file:
            station = extract_stn_info(confs['station_dir'] + file)
            station['id'] = file[0:5]
            if (not (min_lat <= station['lat'] <= max_lat)
                or not (min_lon <= station['lon'] <= max_lon)
                or not station['valid']):
                continue

            content = [str(station['id']), str(station['lat']),
                       str(station['lon']), str(station['height'])]
            writer.writerow(content)

    stations_csv.close()

    # Read wind data
    if not os.path.exists(confs['pickle_dir']):
        os.mkdir(confs['pickle_dir'])
    stations_csv = pd.read_csv(stations_csv_name)

    with open(confs['var_dir'] + 'station_year.pkl', 'rb') as fr:
        station_year = pickle.load(fr)

    for index, id in enumerate(stations_csv['id']):
        years = station_year[id]

        for year in years:
            pickle_name = id + '_' + year + '.pkl'
            if os.path.exists(confs['pickle_dir'] + pickle_name):
                continue

            gzip_file_name = id + 'c' + year + '.txt.gz'
            gzip_file_path = confs['cwind_dir'] + gzip_file_name
            try:
                # print('Processing ' + pickle_name)
                with gzip.GzipFile(gzip_file_path, 'rb') as gz:
                    cwind_text = gz.read()
            except FileNotFoundError as e:
                print(str(e))
                continue
            except EOFError as e:
                print(str(e) + ': ' + gzip_file_name)
                exit(0)
            else:
                print('Processing ' + pickle_name)
            with open(gzip_file_name[0:-3], 'wb') as txt:
                txt.write(cwind_text)

            data_type = {'names': ('year', 'month', 'day', 'hour',
                                   'minute', 'direction', 'speed'),
                         'formats': ('S4', 'i2', 'i2', 'i2', 'i2',
                                     'i4', 'f4')}
            data = np.loadtxt(gzip_file_name[0:-3], skiprows=1,
                              usecols=(0,1,2,3,4,5,6), dtype=data_type)
            os.remove(gzip_file_name[0:-3])

            cwind_1_year = []
            for row in data:
                # Drop possibly existing data records in the end of
                # last year and may LEAD TO UNCOMPLETE DATA
                if bytes.decode(row['year'])[-2:] != year[-2:]:
                    continue
                cwind_10_mins = {}
                cwind_10_mins['height'] = float(
                    stations_csv['height'][index])
                cwind_10_mins['lat'] = float(
                    stations_csv['lat'][index])
                cwind_10_mins['lon'] = float(
                    stations_csv['lon'][index])
                cwind_10_mins['month'] = int(row['month'])
                cwind_10_mins['day'] = int(row['day'])
                cwind_10_mins['time'] = (int(row['hour']) * 60
                                      + int(row['minute']))
                cwind_10_mins['wdir'] = float(row['direction'])
                cwind_10_mins['wspd'] = convert_10(
                    float(row['speed']), cwind_10_mins['height'])
                cwind_1_year.append(cwind_10_mins)

            pickle_file = open(confs['pickle_dir'] + pickle_name, 'wb')
            pickle.dump(cwind_1_year, pickle_file)
            pickle_file.close()
