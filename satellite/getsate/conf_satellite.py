# !/usr/bin/env python

from datetime import date

def configure():
    confs = {}

    rss_data_url = 'http://data.remss.com/'
    data_root_dir = '../sate_data/'
    confs['data_name_len'] = 26
    confs['spec_confs'] = []

    ascat = {}
    ascat['name'] = 'ascat'
    ascat['data_dir'] = data_root_dir + 'ascat/'
    ascat['file_suffix'] = '_v02.1.gz'
    ascat['missing_dates'] = []
    ascat['data_url'] = rss_data_url + 'ascat/metopa/bmaps_v02.1/'
    confs['spec_confs'].append(ascat)

    qscat = {}
    qscat['name'] = 'qscat'
    qscat['data_dir'] = data_root_dir + 'qscat/'
    qscat['file_suffix'] = 'v4.gz'
    qscat['missing_dates'] = [
        date(2000, 11, 17), date(2001, 5, 12), date(2001, 5, 13),
        date(2001, 11, 18), date(2001, 7, 8), date(2002, 3, 20),
        date(2002, 11, 19)
    ]
    qscat['data_url'] = rss_data_url + 'qscat/bmaps_v04/'
    confs['spec_confs'].append(qscat)

    windsat = {}
    windsat['name'] = 'wsat'
    windsat['data_dir'] = data_root_dir + 'windsat/'
    windsat['file_suffix'] = 'v7.0.1.gz'
    windsat['missing_dates'] = []
    windsat['data_url'] = rss_data_url + 'windsat/bmaps_v07.0.1/'
    confs['spec_confs'].append(windsat)

    confs['choose_satellite_prompt'] = \
            ('\nChoose satellite and enter corresponding number:\n'
             + '1. ASCAT\n'
             + '2. QucikSCAT\n'
             + '3. WindSat\n'
             + '\nEnter number of satellite: ')
    confs['input_date_prompt'] = [
        ('\nInputting range of date'
         + '\n\nEnter start date in form of YEAR/MONTH/DAY: '),
        '\nEnter end date in form of YEAR/MONTH/DAY: '
    ]


    return confs
