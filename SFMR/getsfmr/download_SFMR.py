# !/usr/bin/env python

import urllib.request
import re
import requests
import os
import signal
import sys
import pickle

from bs4 import BeautifulSoup
import progressbar

import conf_SFMR

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
    print('Remove ' + current_download_file)
    os.remove(current_download_file)
    # Print log
    print('\nForce quit on %s.\n' % signum)
    # Force quit
    sys.exit(1)

def input_filter(input):
    hurr_year = {}
    mode = 'direct'
    input = input.replace(' ', '').lower()
    parts = re.findall('\[(.*?)\]', input)
    for part in parts:
        if not part.count(','):
            hurr_year[part] = None
            mode = 'indirect'
        else:
            hurr = part.split(',')[1]
            year = part.split(',')[0]
            hurr_year[hurr] = year
    return hurr_year, mode

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

def download_netcdf(hurr_year, mode, confs):
    suffix = '.nc'
    save_root_dir = confs['hurr_dir']
    os.makedirs(save_root_dir, exist_ok=True)

    if mode == 'direct':
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
                    if os.path.exists(file_path):
                        continue
                    global current_download_file
                    current_download_file = file_path
                    global format_custom_text
                    format_custom_text.update_mapping(f=file_name)
                    urllib.request.urlretrieve(href, file_path,
                                               show_progress)

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
    confs = conf_SFMR.configure()
    print(confs['input_prompt'], end='')
    hurr_year, mode = input_filter(input())
    save_hurr_year(confs['var_dir'] + 'hurr_year.pkl', hurr_year)

    print('\nDownloading Hurrican SFMR Data')
    set_format_custom_text(confs['data_name_len'])
    download_netcdf(hurr_year, mode, confs)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGHUP, handler)
    signal.signal(signal.SIGTERM, handler)
    # Original author's target hurricane SFMR data:
    # [2011, Dora] [2014, Simon] [2015, Guillermo]
    # [2015, Patricia] [2016, Javier]
    main()
