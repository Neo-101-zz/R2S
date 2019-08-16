# !/usr/bin/env python

import os
import signal
import sys
from urllib import request
import requests

import progressbar

# Global variables
pbar = None
format_custom_text = None
current_file = None

def arrange_signal():
    """Arrange handler for several signal.
    
    Parameters
    ----------
    None
        Nothing required by this function.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGHUP, handler)
    signal.signal(signal.SIGTERM, handler)

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
    os.remove(current_file)
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

def url_exists(url):
    """Check if url exists.

    Parameters
    ----------
    url : str
        Complete url of file.
    content_type: str, optional
        Content type of file's request header. Default value is content
        type of gzip file.

    Returns
    -------
    Bool
        True if url exists, otherwise False.

    """
    if url.startswith('http'):
        req = requests.head(url)
        if url.endswith('.gz'):
            if req.headers['Content-Type'] == 'application/x-gzip':
                return True
            else:
                return False
        # elif url.endswith('.nc'):
    else:
        # if url.startswith('ftp'):
        return True

def check_period(date_, period):
    start = period[0]
    end = period[1]
    if date_ < start or date_ > end or start > end:
        return False

    return True

def download(url, path):
    """Download file by its url.

    Parameters
    ----------
    url : str
        Complete url of file.
    path : str
        Absolute or saving relative path of file.

    Returns
    -------
    None
        Nothing returned by this function.

    """
    if os.path.exists(path):
        return

    if not url_exists(url):
        print('File doesn\'t exist: ' + url)
        return

    global current_file
    current_file = path

    global format_custom_text
    file_name = path.split('/')[-1]
    format_custom_text.update_mapping(f=file_name)
    request.urlretrieve(url, path, show_progress)
