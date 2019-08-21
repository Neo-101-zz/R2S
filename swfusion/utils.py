from datetime import date
import os
import signal
import sys
import pickle

from urllib import request
import requests
import progressbar
import mysql.connector
from mysql.connector import errorcode

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
                progressbar.Percentage(),
                ' ', progressbar.Bar(marker=progressbar.RotatingMarker()),
                ' ', progressbar.ETA(),
                ' ', progressbar.FileTransferSpeed(),
                # progressbar.Bar(marker=u'\u2588', fill='.',
                #                 left='| ', right= ' |'),
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
    Bool
        True if actually download file, False otherwise.

    """
    if os.path.exists(path):
        return False

    if not url_exists(url):
        print('File doesn\'t exist: ' + url)
        return False

    global current_file
    current_file = path

    global format_custom_text
    file_name = path.split('/')[-1]
    format_custom_text.update_mapping(f=file_name)
    request.urlretrieve(url, path, show_progress)

    return True

def check_period(period, limit, prompt):
    """Check whether period is in range of date limit,
    and correct it if possible.

    """
    start_date = period[0]
    end_date = period[1]
    start_limit = limit['start']
    end_limit = limit['end']

    if end_limit.year == 9999:
        end_limit = date.today()
    # Time flow backward, no chance to correct
    if start_date > end_date:
        print(prompt['error']['time_flow_backward'])
        return False, period
    # Correct start_date to start_limit
    if start_date < start_limit:
        print((prompt['error']['too_early']
               + start_limit.strftime(' %Y/%m/%d.\n')))
        start_date = start_limit
    # Correct end_date to end_limit
    if end_date > end_limit:
        print((prompt['error']['too_late']
               + end_limit.strftime(' %Y/%m/%d.\n')))
        end_date = end_limit

    return True, [start_date, end_date]

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


def input_period(CONFIG):
    print(CONFIG['workflow']['prompt']['info']['period'])
    start_date = filter_date(
        input(CONFIG['workflow']['prompt']['input']['start_date']))
    end_date = filter_date(
        input(CONFIG['workflow']['prompt']['input']['end_date']))

    return [start_date, end_date]

def input_region(CONFIG):
    """Set selected region with user's input of range of latitude and
    longitude.

    """
    print(CONFIG['workflow']['prompt']['info']['specify_region'])
    default_region = [CONFIG['cwind']['default_values']['min_latitude'],
                      CONFIG['cwind']['default_values']['max_latitude'],
                      CONFIG['cwind']['default_values']['min_longitude'],
                      CONFIG['cwind']['default_values']['max_longitude']]

    input_region = [
        input(CONFIG['workflow']['prompt']['input']['min_latitude']),
        input(CONFIG['workflow']['prompt']['input']['max_latitude']),
        input(CONFIG['workflow']['prompt']['input']['min_longitude']),
        input(CONFIG['workflow']['prompt']['input']['max_longitude'])
    ]

    for index, value in enumerate(input_region):
        if len(value) == 0:
            input_region[index] = default_region[index]

    return input_region

def filter_date(input):
    """Filter the inputted date.

    Parameters
    ----------
    input : str
    Inputted string of date in the form of YEAR/MONTH/DAY.

    Returns
    -------
    date
    An idealized naive date in the form of current Gregorian
    calendar.

    """
    year, month, day = input.split('/')
    while month.startswith('0'):
        month = month[1:]
        while day.startswith('0'):
            day = day[1:]

    return date(int(year), int(month), int(day))

def create_database(cnx, db_name):
    """Create a database.

    """
    cursor = cnx.cursor()
    try:
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS {} DEFAULT CHARACTER SET " \
            "'utf8'".format(db_name))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)

def use_database(cnx, db_name):
    """Swith to particular database.

    """
    cursor = cnx.cursor()
    try:
        cursor.execute("USE {}".format(db_name))
    except mysql.connector.Error as err:
        print("Database {} does not exists.".format(db_name))
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            create_database(cursor, db_name)
            print("Database {} created successfully.".format(db_name))
            cnx.database = db_name
        else:
            print(err)
            exit(1)
