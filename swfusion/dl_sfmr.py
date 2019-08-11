# !/usr/bin/env python
"""Download SFMR data of hurricane from NOAA HRD.

"""
import re
import requests
import os
import pickle

from bs4 import BeautifulSoup

import conf_sfmr
import dl_util

def input_filter(input):
    """Filter input to get a dict of hurricane name and year.

    Parameters
    ----------
    input : str
        Inputted string.

    Returns
    -------
    hurr_year : dict
        A dict with hurricane name as key and its year as value.

    """
    hurr_year = {}
    input = input.replace(' ', '').lower()
    parts = re.findall('\[(.*?)\]', input)
    for part in parts:
        if part.count(','):
            hurr = part.split(',')[1]
            year = part.split(',')[0]
            hurr_year[hurr] = year
        else:
            print('Input format error.')
            exit(0)
    return hurr_year

def dl_sfmr(hurr_year, confs):
    """Download SFMR data of hurricanes.

    Parameters
    ----------
    hurr_year : dict
        A dict with hurricane name as key and its year as value.
    confs : dict
        A dict of configuration.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    suffix = '.nc'
    save_root_dir = confs['hurr_dir']
    os.makedirs(save_root_dir, exist_ok=True)

    for hurr in hurr_year.keys():
        # Create directory to store SFMR files
        os.makedirs(save_root_dir + hurr_year[hurr],
                    exist_ok=True)
        dir_path = (save_root_dir + hurr_year[hurr]
                    + '/' + hurr + '/')
        os.makedirs(dir_path, exist_ok=True)
        # Generate keyword to consist url
        keyword = hurr + hurr_year[hurr]
        url = '{0}{1}{2}'.format(
            confs['storm_SFMR_url_prefix'],
            keyword,
            confs['storm_SFMR_url_suffix'])
        # Get page according to url
        page = requests.get(url)
        data = page.text
        soup = BeautifulSoup(data, features='lxml')
        anchors = soup.find_all('a')

        for link in anchors:
            href = link.get('href')
            # Find href of netcdf file
            if href.endswith(suffix):
                # Extract file name
                file_name = href.split('/')[-1]
                file_path = dir_path + file_name
                dl_util.download(href, file_path)

def save_hurr_year(path, var):
    """Append hurr_year to the pickle file which store this variable.
    
    Parameters
    ----------
    path : str
        Location of pickle file to store the variable.
    var : dict
        Store the relation in the form of dict, of which key is
        hurricane name (str) and value is corresponding year (str).
    
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
            if key in rel and rel[key] != var[key]:
                print('Error: two versions of hurr_year has conflicted:')
                print('old version: ' + key + '-' + rel[key])
                print('new version: ' + key + '-' + var[key])
                exit(0)
            else:
                rel[key] = var[key]
    else:
        rel = var

    with open(path, 'wb') as fw:
        pickle.dump(rel, fw)

def main():
    confs = conf_sfmr.configure()
    print(confs['input_prompt'], end='')
    hurr_year = input_filter(input())
    save_hurr_year(confs['var_dir'] + 'hurr_year.pkl', hurr_year)

    print('\nDownloading Hurrican SFMR Data')
    dl_util.set_format_custom_text(confs['data_name_len'])
    dl_sfmr(hurr_year, confs)
    print()

if __name__ == '__main__':
    dl_util.arrange_signal()
    # Original author's target hurricane SFMR data:
    # [2011, Dora] [2014, Simon] [2015, Guillermo]
    # [2015, Patricia] [2016, Javier]
    main()
