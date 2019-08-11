# !/usr/bin/env python

from datetime import date

def configure():
    confs = {}

    rss_data_url = 'http://data.remss.com/'
    data_root_dir = '../data/satel/'
    confs['data_name_len'] = 26
    confs['spec_confs'] = []

    ascat = {}
    ascat['name'] = 'ascat'
    ascat['data_dir'] = data_root_dir + 'ascat/'
    ascat['file_suffix'] = '_v02.1.gz'
    ascat['missing_dates'] = [
        date(2007, 4, 21), date(2007, 4, 22), date(2007, 4, 23),
        date(2007, 4, 24), date(2007, 9, 18), date(2008, 1, 17),
        date(2008, 3, 20), date(2011, 5, 15)
    ]
    ascat['missing_dates_file'] = (data_root_dir 
                                   + 'ascat_missing_dates.pkl')
    ascat['missing_value'] = -999.0
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
    windsat['missing_dates'] = [
        date(2004, 1, 28), date(2004, 1, 30), date(2004, 3, 1),
        date(2004, 3, 2), date(2004, 3, 3), date(2004, 4, 30)
    ]
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
