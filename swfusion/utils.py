from datetime import date
import datetime
import logging
import math
import os
import signal
import sys
import pickle
import time

import numpy as np
from urllib import request
import requests
import progressbar
import mysql.connector
from mysql.connector import errorcode
import netCDF4
import bytemaps
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Integer, Float, String, DateTime, Boolean
from sqlalchemy import Table, Column, MetaData
from sqlalchemy.orm import mapper
from sqlalchemy import tuple_
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import pygrib
from global_land_mask import globe
from scipy import interpolate
import pandas as pd
from geopy import distance

from amsr2_daily import AMSR2daily
from ascat_daily import ASCATDaily
from quikscat_daily_v4 import QuikScatDaily
from windsat_daily_v7 import WindSatDaily

# Global variables
logger = logging.getLogger(__name__)
pbar = None
format_custom_text = None
current_file = None

DEGREE_OF_ONE_NMILE = float(1)/60
KM_OF_ONE_NMILE = 1.852
KM_OF_ONE_DEGREE = KM_OF_ONE_NMILE / DEGREE_OF_ONE_NMILE
RADII_LEVELS = [34, 50, 64]


# Python program to check if rectangles overlap 
class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 


# Returns true if two rectangles(l1, r1)
# and (l2, r2) overlap 
def doOverlap(l1, r1, l2, r2):
    """Check if two rectangles overlap.

    Parameters
    ----------
    l1: Point
        The left-top corner of the first rectangle.
    r1: Point
        The right-bottom corner of the first rectangle.
    l2: Point
        The left-top corner of the second rectangle.
    r2: Point
        The right-bottom corner of the second rectangle.

    Returns
    -------
    bool
        True if two rectangles overlap, False otherwise.

    """
    # If one rectangle is on left side of other 
    if(l1.x > r2.x or l2.x > r1.x): 
        return False

    # If one rectangle is above other 
    if(l1.y < r2.y or l2.y < r1.y): 
        return False

    return True


def delete_last_lines(n=1):
    CURSOR_UP_ONE = '\x1b[1A'
    CURSOR_LEFT_HEAD = '\x1b[1G'
    ERASE_LINE = '\x1b[2K'

    for _ in range(n):
        sys.stdout.write(ERASE_LINE)
        sys.stdout.write(CURSOR_LEFT_HEAD)


def setup_signal_handler():
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


def reset_signal_handler():
    global current_file
    current_file = None

    signal.signal(signal.SIGINT, signal.default_int_handler)


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
    if current_file is not None:
        os.remove(current_file)
        info = f'Removing uncompleted downloaded file: {current_file}'
        logger.info(info)
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
                ' | %-8s' % sizeof_fmt(total_size),
                ' | ', progressbar.Percentage(),
                ' ', progressbar.Bar(marker='#', left='| ', right=' |'),
                ' ', progressbar.ETA(),
                ' | ', progressbar.FileTransferSpeed(),
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
        key = 'Content-Type'
        req = requests.head(url)
        # Works for '.gz' file and '.nc' file
        try:
            if key not in req.headers:
                return False
            if req.headers[key].startswith('application'):
                return True
            else:
                return False
        except Exception as msg:
            breakpoint()
            exit(msg)
    else:
        # if url.startswith('ftp'):
        return True

def check_period(temporal, period):
    if type(temporal) is datetime.date:
        period = [x.date() for x in period]
    elif type(temporal) is datetime.time:
        period = [x.time() for x in period]
    elif type(temporal) is not datetime.datetime:
        logger.error('Type of inputted temporal variable should be ' \
             'datetime.date or datetime.time or datetime.datetime')
    start = period[0]
    end = period[1]
    if temporal < start or temporal > end or start > end:
        return False

    return True

def download(url, path, progress=False):
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
        return

    if not url_exists(url):
        print('File doesn\'t exist: ' + url)
        return

    global current_file
    current_file = path

    global format_custom_text
    file_name = path.split('/')[-1]
    if format_custom_text is not None:
        format_custom_text.update_mapping(f=file_name)
    try:
        if progress:
            request.urlretrieve(url, path, show_progress)
        else:
            request.urlretrieve(url, path)
    except Exception as msg:
        logger.exception(f'Error occured when downloading {path} from {url}')

def check_and_update_period(period, limit, prompt):
    """Check whether period is in range of date limit,
    and correct it if possible. Used in processing inputted period.

    """
    start_datetime = period[0]
    end_datetime = period[1]
    start_limit = limit['start']
    end_limit = limit['end']

    # Time flow backward, no chance to correct
    if start_datetime > end_datetime:
        print(prompt['error']['time_flow_backward'])
        return False, period
    if end_limit.year == 9999:
        end_limit = datetime.datetime.now()
    # Correct start_datetime to start_limit
    if start_datetime < start_limit:
        print('%s %s' % (prompt['error']['too_early'], str(start_limit)))
        start_datetime = start_limit
    # Correct end_datetime to end_limit
    if end_datetime > end_limit:
        print('%s %s' % (prompt['error']['too_late'], str(end_limit)))
        end_datetime = end_limit

    return True, [start_datetime, end_datetime]

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
    start_datetime = filter_datetime(
        input(CONFIG['workflow']['prompt']['input']['start_date']))
    end_datetime = filter_datetime(
        input(CONFIG['workflow']['prompt']['input']['end_date']))

    return [start_datetime, end_datetime]

def input_region(CONFIG):
    """Set selected region with user's input of range of latitude and
    longitude.

    """
    print(CONFIG['workflow']['prompt']['info']['specify_region'])
    default_region = [
        CONFIG['workflow']['default_region']['min_latitude'],
        CONFIG['workflow']['default_region']['max_latitude'],
        CONFIG['workflow']['default_region']['min_longitude'],
        CONFIG['workflow']['default_region']['max_longitude']
    ]

    inputted_region = [
        input(CONFIG['workflow']['prompt']['input']['min_latitude']),
        input(CONFIG['workflow']['prompt']['input']['max_latitude']),
        input(CONFIG['workflow']['prompt']['input']['min_longitude']),
        input(CONFIG['workflow']['prompt']['input']['max_longitude'])
    ]

    for index, value in enumerate(inputted_region):
        if len(value) == 0:
            inputted_region[index] = default_region[index]
        else:
            inputted_region[index] = float(inputted_region[index])

    return inputted_region

def filter_datetime(input):
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
    return datetime.datetime.strptime(input, '%Y/%m/%d/%H/%M/%S')

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
    return datetime.datetime.strptime(input + '/0/0/0',
                               '%Y/%m/%d/%H/%M/%S').date()

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

def extract_netcdf_to_table(nc_file, table_class, skip_vars,
                            datetime_func, datetime_col_name, missing,
                            valid_func, unique_func, unique_col_name,
                            lat_name, lon_name,
                            period, region, not_null_vars):
    """Extract variables from netcdf file to generate an instance of
    table class.

    Paramters
    ---------
    nc_file : str
        Path of netcdf file.
    table_class :
        The class which represents data of netcdf file.
    skip_vars : list of str
        Variables' names that to be skipped reading.
    datetime_func : func
        Function which maps temporal variables of netcdf file to datetime
        attribute of table class.
    datetime_col_name: str
        Name of table column which represents datetime.
    missing : custom
        Could be numpy.ma.core.masked or something else that user custom.
    valid_func : func
        Function which check whether a row of table is valid or not.
    unique_func : func
        Function which returns unique column value.
    unique_col_name : str
        Name of table column which is unique.
    lat_name : str
        Name of table column which represents latitude.
    lon_name : str
        Name of table column which represents longitude.
    period : list of datetime
        User-specified range of datetime.  Length is two.  Fisrt element is
        start datetime and second element is end datetime.
    region : list of float

    Returns
    -------
    table_row : instance of `table_class`
        Instance of table class that has data of netcdf file as its
        attributes.

    """
    dataset = netCDF4.Dataset(nc_file)
    vars = dataset.variables
    if 'time' not in dataset.dimensions.keys():
        exit('[Error] NetCDF dataset does not have "time" dimension')
    length = dataset.dimensions['time'].size
    min_lat, max_lat = region[0], region[1]
    min_lon, max_lon = region[2], region[3]

    # Store all rows
    whole_table = []
    ds_min_lat = 90.0
    ds_max_lat = -90.0
    ds_min_lon = 360.0
    ds_max_lon = 0.0

    for i in range(length):
        table_row = table_class()
        # Check whether the row is valid or not
        if not valid_func(vars, i):
            continue
        # Region check
        lat = vars[lat_name][i]
        lon = (vars[lon_name][i] + 360) % 360
        if (not lat or not lon or lat < min_lat or lat > max_lat
            or lon < min_lon or lon > max_lon):
            continue
        setattr(table_row, lat_name, convert_dtype(lat))
        setattr(table_row, lon_name, convert_dtype(lon))
        skip_vars.append(lat_name)
        skip_vars.append(lon_name)
        # Set datetime
        try:
            datetime_ = datetime_func(vars, i, missing)
        except Exception as msg:
            breakpoint()
            exit(msg)
        # Period check
        if not datetime_ or not check_period(datetime_, period):
            continue
        setattr(table_row, datetime_col_name, datetime_)

        setattr(table_row, unique_col_name, unique_func(datetime_, lat, lon))

        valid = True
        # Process all variables of NetCDF dataset
        for var_name in vars.keys():
            # Skip specified variables
            if var_name in skip_vars:
                continue
            # Set columns which is not nullable
            if var_name in not_null_vars:
                # Skip this row if not null variable is null
                if is_missing(vars[var_name][i],  missing):
                    valid = False
                    break
                setattr(table_row, var_name,
                        convert_dtype(vars[var_name][i]))
                continue
            # Set columns which is nullable
            setattr(table_row, var_name, convert_dtype(vars[var_name][i]))

        if valid:
            whole_table.append(table_row)

            if getattr(table_row, lat_name) < ds_min_lat:
                ds_min_lat = getattr(table_row, lat_name)
            if getattr(table_row, lat_name) > ds_max_lat:
                ds_max_lat = getattr(table_row, lat_name)
            if getattr(table_row, lon_name) < ds_min_lon:
                ds_min_lon = getattr(table_row, lon_name)
            if getattr(table_row, lon_name) > ds_max_lon:
                ds_max_lon = getattr(table_row, lon_name)

    return whole_table, ds_min_lat, ds_max_lat, ds_min_lon, ds_max_lon

def is_missing(var, missing):
    if var is missing:
        return True
    return False

def convert_dtype(nparray):
    # In case that type of array is <class 'numpy.ma.core.MaskedArray'>
    if nparray is np.ma.core.masked:
        return None
    for dtype in [np.int32, np.int64]:
        if np.issubdtype(nparray.dtype, dtype):
            return int(nparray)
    for dtype in [np.float32, np.float64]:
        if np.issubdtype(nparray.dtype, dtype):
            return float(nparray)
    if np.issubdtype(nparray.dtype, np.dtype(bool).type):
        return bool(nparray)

def create_table_from_netcdf(engine, nc_file, table_name, session,
                             skip_vars=None, notnull_vars=None,
                             unique_vars=None, custom_cols=None):
    class Netcdf(object):
        pass

    if engine.dialect.has_table(engine, table_name): 
        metadata = MetaData(bind=engine, reflect=True)
        t = metadata.tables[table_name]
        mapper(Netcdf, t)

        return Netcdf

    # Sort custom_cols by column indices
    tmp = custom_cols
    custom_cols = dict()
    for i in sorted(tmp.keys()):
        custom_cols[i] = tmp[i]

    dataset = netCDF4.Dataset(nc_file)
    vars = dataset.variables

    cols = []
    key = Column('key', Integer(), primary_key=True)
    cols.append(key)
    index = 1
    for var_name in vars.keys():
        col_name = var_name.replace('-', '_')
        while index in custom_cols:
            cols.append(custom_cols[index])
            index += 1
        if var_name in skip_vars:
            continue
        nullable = True if var_name not in notnull_vars else False
        unique = False if var_name not in unique_vars else True
        try:
            cols.append(var2sacol(vars, var_name, col_name, nullable, unique))
        except Exception as msg:
            breakpoint()
            exit(msg)
        index += 1
    # In case that there is one custom col to insert into tail of row
    if index in custom_cols:
        cols.append(custom_cols[index])
        index += 1
    metadata = MetaData(bind=engine)
    t = Table(table_name, metadata, *cols)
    metadata.create_all()
    mapper(Netcdf, t)
    session.commit()

    return Netcdf

def table_objects_same(object_1, object_2, unique_cols):
    unique_key = '_sa_instance_state'
    for key in unique_cols:
        if object_1.__dict__[key] == object_2.__dict__[key]:
            return True

    return False

def bulk_insert_avoid_duplicate_unique(total_sample, batch_size,
                                       table_class, unique_cols,
                                       session, check_self=False):
    """
    Bulkly insert into a table which has unique columns.

    """
    total = len(total_sample)
    count = 0

    while total_sample:
        # count += batch_size
        # progress = float(count)/total*100
        # print('\r{:.1f}%'.format(progress), end='')

        batch = total_sample[:batch_size]
        total_sample = total_sample[batch_size:]

        if check_self:
            # Remove duplicate table class objects in batch
            batch_len = len(batch)
            dup_val = -999
            try:
                for i in range(batch_len-1):
                    for j in range(i+1, batch_len):
                        if batch[i] == dup_val or batch[j] == dup_val:
                            continue
                        if table_objects_same(batch[i], batch[j],
                                              unique_cols):
                            batch[j] = dup_val
            except Exception as msg:
                breakpoint()
                exit(msg)
            old_batch = batch
            batch = []
            for i in range(batch_len):
                if old_batch[i] != -999:
                    batch.append(old_batch[i])

        existing_records = dict(
            (
                tuple([getattr(data, name) for name in unique_cols]),
                data
            )
            for data in session.query(table_class).filter(
                tuple_(*[getattr(table_class, name) \
                         for name in unique_cols]).in_(
                             [
                                 tuple_(*[getattr(x, name) \
                                 for name in unique_cols]) \
                                 for x in batch
                             ]
                         )
            )
        )

        inserts = []
        for data in batch:
            existing = existing_records.get(
                tuple([getattr(data, name) for name in unique_cols]),
                None)
            if existing:
                pass
            else:
                inserts.append(data)

        try:
            if inserts:
                # session.add_all(inserts)
                session.bulk_insert_mappings(
                    table_class,
                    [
                        row2dict(record)
                        for record in inserts
                    ],
                )
        except Exception as msg:
            breakpoint()
            exit(msg)

    session.commit()

def row2dict(row):
    d = row.__dict__
    d.pop('_sa_instance_state', None)

    return d

def create_table_from_bytemap(engine, satel_name, bytemap_file, table_name,
                              session, skip_vars=None, notnull_vars=None,
                              unique_vars=None, custom_cols=None):

    class Satel(object):
        pass

    if engine.dialect.has_table(engine, table_name): 
        metadata = MetaData(bind=engine, reflect=True)
        t = metadata.tables[table_name]
        mapper(Satel, t)

        return Satel

    # Sort custom_cols by column indices
    tmp = custom_cols
    custom_cols = dict()
    for i in sorted(tmp.keys()):
        custom_cols[i] = tmp[i]

    dataset = dataset_of_daily_satel(satel_name, bytemap_file)
    vars = dataset.variables

    cols = []
    key = Column('key', Integer(), primary_key=True)
    cols.append(key)
    index = 1
    for var_name in vars.keys():
        col_name = var_name.replace('-', '_')
        while index in custom_cols:
            cols.append(custom_cols[index])
            index += 1
        if var_name in skip_vars:
            continue
        nullable = True if var_name not in notnull_vars else False
        unique = False if var_name not in unique_vars else True
        cols.append(var2sacol(vars, var_name, col_name, nullable, unique))
        index += 1
    # In case that there is one custom col to insert into tial of row
    if -1 in custom_cols:
        cols.append(custom_cols[-1])

    metadata = MetaData(bind=engine)
    t = Table(table_name, metadata, *cols)
    metadata.create_all()
    mapper(Satel, t)
    session.commit()

    return Satel

def var2sacol(vars, var_name, col_name, nullable=True, unique=False):
    var_dtype = vars[var_name].dtype
    column = None
    for dtype in [np.int32, np.int64]:
        if np.issubdtype(var_dtype, dtype):
            return Column(col_name, Integer(), nullable=nullable,
                          unique=unique)
    for dtype in [np.float32, np.float64]:
        if np.issubdtype(var_dtype, dtype):
            return Column(col_name, Float(), nullable=nullable,
                          unique=unique)
    if np.issubdtype(var_dtype, np.dtype(bool).type):
        return Column(col_name, Boolean(), nullable=nullable,
                      unique=unique)

def dataset_of_daily_satel(satel_name, file_path, missing_val=-999.0):
    if satel_name == 'ascat':
        dataset = ASCATDaily(file_path, missing=missing_val)
    elif satel_name == 'qscat':
        dataset = QuikScatDaily(file_path, missing=missing_val)
    elif satel_name == 'wsat':
        dataset = WindSatDaily(file_path, missing=missing_val)
    elif satel_name == 'amsr2':
        dataset = AMSR2daily(file_path, missing=missing_val)
    else:
        sys.exit('Invalid satellite name')

    if not dataset.variables:
        sys.exit('[Error] File not found: ' + file_path)

    return dataset

def show_bytemap_dimensions(ds):
    print('')
    print('Dimensions')
    for dim in ds.dimensions:
        aline = ' '.join([' '*3, dim, ':', str(ds.dimensions[dim])])
        print(aline)

def show_bytemap_variables(ds):
    print('')
    print('Variables:')
    for var in ds.variables:
        aline = ' '.join([' '*3, var, ':', ds.variables[var].long_name])
        print(aline)

def show_bytemap_validrange(ds):
    print('')
    print('Valid min and max and units:')
    for var in ds.variables:
        aline = ' '.join([' '*3, var, ':',
                str(ds.variables[var].valid_min), 'to',
                str(ds.variables[var].valid_max),
                '(',ds.variables[var].units,')'])
        print(aline)

def find_index(range, lat_or_lon):
    # latitude: from -89.875 to 89.875, 720 values, interval = 0.25
    # longitude: from 0.125 to 359.875, 1440 values, interval = 0.25
    res = []
    for idx, val in enumerate(range):
        if lat_or_lon == 'lat':
            delta = val + 89.875
        elif lat_or_lon == 'lon':
            delta = val - 0.125
        else:
            exit('Error parameter lat_or_lon: ' + lat_or_lon)
        intervals = delta / 0.25
        # Find index of min_lat or min_lon
        if not idx:
            res.append(math.ceil(intervals))
        # Find index of max_lat or max_lon
        else:
            res.append(math.floor(intervals))

    # in case that two elements of range is same (representing single
    # point)
    if res[0] > res[1]:
        tmp = res[0]
        res[0] = res[1]
        res[1] = tmp

    return res

def extract_bytemap_to_table(satel_name, bm_file, table_class, skip_vars,
                            datetime_func, datetime_col_name, missing,
                            valid_func, unique_func, unique_col_name,
                            lat_name, lon_name,
                            period, region, not_null_vars):
    """Extract variables from netcdf file to generate an instance of
    table class.

    Paramters
    ---------
    bm_file : str
        Path of bytemap file.
    table_class :
        The class which represents data of netcdf file.
    skip_vars : list of str
        Variables' names that to be skipped reading.
    datetime_func : func
        Function which maps temporal variables of netcdf file to datetime
        attribute of table class.
    datetime_col_name: str
        Name of table column which represents datetime.
    missing : custom
        Could be numpy.ma.core.masked or something else that user custom.
    valid_func : func
        Function which check whether a row of table is valid or not.
    unique_func : func
        Function which returns unique column value.
    unique_col_name : str
        Name of table column which is unique.
    lat_name : str
        Name of table column which represents latitude.
    lon_name : str
        Name of table column which represents longitude.
    period : list of datetime
        User-specified range of datetime.  Length is two.  Fisrt element is
        start datetime and second element is end datetime.
    region : list of float

    Returns
    -------
    table_row : instance of `table_class`
        Instance of table class that has data of netcdf file as its
        attributes.

    """

    dataset = dataset_of_daily_satel(satel_name, bm_file)
    vars = dataset.variables

    min_lat, max_lat = region[0], region[1]
    min_lon, max_lon = region[2], region[3]
    min_lat_idx, max_lat_idx = find_index([min_lat, max_lat], 'lat')
    lat_indices = [x for x in range(min_lat_idx, max_lat_idx+1)]
    min_lon_idx, max_lon_idx = find_index([min_lon, max_lon], 'lon')
    lon_indices = [x for x in range(min_lon_idx, max_lon_idx+1)]

    lat_len = len(lat_indices)
    lon_len = len(lon_indices)
    total = 2 * lat_len * lon_len
    count = 0

    # Store all rows
    whole_table = []

    st = time.time()
    iasc = [0, 1]
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    for i in iasc:
        for j in lat_indices:
            for k in lon_indices:
                count += 1
                if count % 10000 == 0:
                    print('\r{:5f}%'.format((float(count)/total)*100), end='')
                # if j == 120:
                #     et = time.time()
                #     print('\ntime: %s' % (et - st))
                #     breakpoint()
                if not valid_func(vars, i, j, k):
                    continue
                table_row = table_class()
                lat = vars[lat_name][j]
                lon = vars[lon_name][k]
                if (not lat or not lon or lat < min_lat or lat > max_lat
                    or lon < min_lon or lon > max_lon):
                    continue
                setattr(table_row, lat_name, convert_dtype(lat))
                setattr(table_row, lon_name, convert_dtype(lon))
                skip_vars.append(lat_name)
                skip_vars.append(lon_name)
                # Set datetime
                try:
                    datetime_ = datetime_func(bm_file, vars, i, j, k,
                                              missing)
                except Exception as msg:
                    breakpoint()
                    exit(msg)
                # Period check
                if not datetime_ or not check_period(datetime_, period):
                    continue
                setattr(table_row, datetime_col_name, datetime_)

                setattr(table_row, unique_col_name,
                        unique_func(datetime_, lat, lon))

                valid = True
                # Process all variables of NetCDF dataset
                for var_name in vars.keys():
                    # Skip specified variables
                    if var_name in skip_vars:
                        continue
                    # Set columns which is not nullable
                    if var_name in not_null_vars:
                        # Skip this row if not null variable is null
                        if is_missing(vars[var_name][i][j][k],  missing):
                            valid = False
                            break
                        setattr(table_row, var_name,
                                convert_dtype(vars[var_name][i][j][k]))
                        continue
                    # Set columns which is nullable
                    setattr(table_row, var_name,
                            convert_dtype(vars[var_name][i][j][k]))

                if valid:
                    whole_table.append(table_row)

    return whole_table

def gen_space_time_fingerprint(datetime, lat, lon):

    return '%s %f %f' % (datetime, lat, lon)

def cut_map(satel_name, dataset, region, year, month, day,
            missing_val=-999.0):
    min_lat, max_lat = find_index([region[0], region[1]], 'lat')
    lat_indices = [x for x in range(min_lat, max_lat+1)]
    min_lon, max_lon = find_index([region[2], region[3]], 'lon')
    lon_indices = [x for x in range(min_lon, max_lon+1)]

    data_list = []
    iasc = [0, 1]
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    num = len(lat_indices) * len(lon_indices) * 2
    num_c1, num_c2, num_c3 = 0, 0, 0
    vars = dataset.variables
    n_cut_land = 0
    n_cut_missing = 0
    n_cut_wdir = 0

    for i in iasc:
        for j in lat_indices:
            for k in lon_indices:
                cut_missing = vars['nodata'][i][j][k]
                if cut_missing:
                    num_c1 += 1
                    continue
                cut_mingmt = vars['mingmt'][i][j][k]
                cut_land = vars['land'][i][j][k]
                if cut_land:
                    n_cut_land += 1
                if cut_missing:
                    n_cut_missing += 1
                if satel_name == 'ascat' or satel_name == 'qscat':
                    cut_wspd = vars['windspd'][i][j][k]
                    cut_wdir = vars['winddir'][i][j][k]
                    cut_rain = vars['scatflag'][i][j][k]
                elif satel_name == 'wsat':
                    cut_rain = vars['rain'][i][j][k]
                    cut_wspd_lf = vars['w-lf'][i][j][k]
                    cut_wspd_mf = vars['w-mf'][i][j][k]
                    cut_wspd_aw = vars['w-aw'][i][j][k]
                    cut_wdir = vars['wdir'][i][j][k]
                else:
                    sys.exit('satel_name is wrong.')

                if cut_wdir == missing_val:
                    n_cut_wdir += 1
                if cut_missing or cut_land or cut_wdir == missing_val:
                    # same pass condition for all satellites
                    num_c1 += 1
                    continue

                if satel_name == 'ascat' or satel_name == 'qscat':
                    if cut_wspd == missing_val:
                        num_c2 += 1
                        continue
                elif satel_name == 'wsat':
                    if (cut_wspd_lf == missing_val
                        or cut_wspd_mf == missing_val
                        or cut_wspd_aw == missing_val):
                        # at least one of three wind speed is missing
                        num_c3 += 1
                        continue

                data_point = {}
                data_point['iasc'] = i
                data_point['lat'] = vars['latitude'][j]
                data_point['lon'] = vars['longitude'][k]
                data_point['wdir'] = cut_wdir
                data_point['rain'] = cut_rain
                data_point['time'] = cut_mingmt
                data_point['year'] = year
                data_point['month'] = month
                data_point['day'] = day

                if satel_name == 'ascat' or satel_name == 'qscat':
                    data_point['wspd'] = cut_wspd
                elif satel_name == 'wsat':
                    data_point['w-lf'] = cut_wspd_lf
                    data_point['w-mf'] = cut_wspd_mf
                    data_point['w-aw'] = cut_wspd_aw

                data_list.append(data_point)

    print()
    print('total data point: ' + str(num))
    print('cut_missing: ' + str(n_cut_missing))
    print('cut_land: ' + str(n_cut_land))
    print('cut_wdir: ' + str(n_cut_wdir))
    print('skip condition 1: ' + str(num_c1))
    print('skip condition 2: ' + str(num_c2))
    print('skip condition 3: ' + str(num_c3))
    print('returned data point: ' + str(len(data_list)))
   #  print()

    return data_list

def narrow_map(dataset, region):
    # Find rectangle range of area
    min_lat, max_lat = find_index([region[0], region[1]], 'lat')
    min_lon, max_lon = find_index([region[2], region[3]], 'lon')

    map = {}
    map['wspd'] = []
    map['wdir'] = []
    map['rain'] = []
    map['time'] = []
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    iasc = [0, 1]
    vars = dataset.variables
    for i in iasc:
        wspd = vars['windspd'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        wdir = vars['winddir'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        rain = vars['scatflag'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        time = vars['mingmt'][i][min_lat:max_lat+1, min_lon:max_lon+1]
        map['wspd'].append(wspd)
        map['wdir'].append(wdir)
        map['rain'].append(rain)
        map['time'].append(time)

    return map

def get_class_by_tablename(engine, table_fullname):
    """Return class reference mapped to table.

    :param table_fullname: String with fullname of table.
    :return: Class reference or None.
    """
    class Template(object):
        pass

    if engine.dialect.has_table(engine, table_fullname):
        metadata = MetaData(bind=engine, reflect=True)
        t = metadata.tables[table_fullname]
        mapper(Template, t)
        return Template
    else:
        return None

def add_column(engine, table_name, column):
    column_name = column.compile(dialect=engine.dialect)
    column_type = column.type.compile(engine.dialect)
    connection = engine.connect()
    result = connection.execute('ALTER TABLE %s ADD COLUMN %s %s' % (
        table_name, column_name, column_type))
    connection.close()

def setup_database(the_class, Base):
    DB_CONFIG = the_class.CONFIG['database']
    PROMPT = the_class.CONFIG['workflow']['prompt']
    DBAPI = DB_CONFIG['db_api']
    USER = DB_CONFIG['user']
    password_ = the_class.db_root_passwd
    HOST = DB_CONFIG['host']
    PORT = DB_CONFIG['port']
    DB_NAME = DB_CONFIG['db_name']
    ARGS = DB_CONFIG['args']

    try:
        # ATTENTION
        #
        # Before connect to MySQL server, we need to check the way
        # of connection first.
        # According to 'https://dev.mysql.com/doc/refman/8.0/en/'
        # 'can-not-connect-to-server.html',
        # a MySQL client on Unix can connect to the mysqld server in
        # two different ways:
        # 1) By using a Unix socket file to connect
        # through a file in the file system (default /tmp/mysql.sock).
        # 2) By using TCP/IP, which connects through a port number.
        #
        # We can check it by this command:
        # shell> mysqladmin version
        # Maybe need user name and password.
        #
        # If we are connecting to mysqld server by using a Unix socket,
        # there should not be 'host' and 'port' parameters in the
        # function mysql.connector.connect()
        #
        # If we are connecting to mysqld server by using TCP/IP,
        # 'host' and 'port' are needed.
        the_class.cnx = mysql.connector.connect(
            user=USER, password=password_,
            # host=HOST, port=PORT,
            use_pure=True)
        the_class.cursor = the_class.cnx.cursor()
        create_database(the_class.cnx, DB_NAME)
        use_database(the_class.cnx, DB_NAME)
    except Exception as msg:
        breakpoint()
        exit(msg)

    # ATTENTION
    # According to docs of SQLAlchemy, we would better not to
    # use MySQL Connector/Python as DBAPI.
    # The MySQL Connector/Python DBAPI has had many issues
    # since its release, some of which may remain unresolved,
    # and the mysqlconnector dialect is not tested as part of
    # SQLAlchemy’s continuous integration.
    # The recommended MySQL dialects are mysqlclient and PyMySQL.
    # Reference: 'https://docs.sqlalchemy.org/en/13/dialects/'
    # 'mysql.html#module-sqlalchemy.dialects.mysql.mysqlconnector'

    # Define the MySQL engine using mysqlclient DBAPI
    connect_string = (f"""{DBAPI}://{USER}:{password_}@{HOST}"""
                      f"""/{DB_NAME}""")
    the_class.engine = create_engine(connect_string, echo=False)
    # Create table of the class
    Base.metadata.create_all(the_class.engine)
    the_class.Session = sessionmaker(bind=the_class.engine)
    the_class.session = the_class.Session()

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

def get_subset_range_of_grib_point(lat, lon, lat_grid_points,
                                   lon_grid_points):
    lon = (lon + 360) % 360

    lat_ae = [abs(lat-y) for y in lat_grid_points]
    lon_ae = [abs(lon-x) for x in lon_grid_points]

    lat_match = lat_grid_points[lat_ae.index(min(lat_ae))]
    lon_match = lon_grid_points[lon_ae.index(min(lon_ae))]

    lat1 = lat_match if lat > lat_match else lat
    lat2 = lat_match if lat < lat_match else lat
    lon1 = lon_match if lon > lon_match else lon
    lon2 = lon_match if lon < lon_match else lon

    return lat1, lat2, lon1, lon2

def get_latlon_index_of_closest_grib_point(lat, lon, lat_grid_points,
                                           lon_grid_points):
    lon = (lon + 360) % 360

    lat_ae = [abs(lat-y) for y in lat_grid_points]
    lon_ae = [abs(lon-x) for x in lon_grid_points]

    lat_match_index = lat_ae.index(min(lat_ae))
    lon_match_index = lon_ae.index(min(lon_ae))

    return lat_match_index, lon_match_index

def get_subset_range_of_grib(lat, lon, lat_grid_points, lon_grid_points,
                             edge, mode='rss', spatial_resolution=None):
    lon = (lon + 360) % 360

    lat_ae = [abs(lat-y) for y in lat_grid_points]
    lon_ae = [abs(lon-x) for x in lon_grid_points]

    lat_match = lat_grid_points[lat_ae.index(min(lat_ae))]
    lon_match = lon_grid_points[lon_ae.index(min(lon_ae))]

    half_edge = float(edge / 2)

    if lat_match - half_edge < -90 or lat_match + half_edge > 90 :
        return False, 0, 0, 0, 0

    lat1 = lat_match - half_edge
    lat2 = lat_match + half_edge
    lon1 = (lon_match - half_edge + 360) % 360
    lon2 = (lon_match + half_edge + 360) % 360

    # When the edge of square along parallel crosses the primie meridian
    if lon2 - lon1 != 2 * half_edge:
        return False, None, None, None, None

    if mode == 'era5':
        lat2 += spatial_resolution
        lon2 += spatial_resolution

    return True, lat1, lat2, lon1, lon2

def area_of_contour(vs):
    """Use Green's theorem to compute the area enclosed by the given
    contour.

    """
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its
    height.

    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2,
                        height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + datetime.timedelta(hours=t.minute//30))

def draw_compare_basemap(ax, lon1, lon2, lat1, lat2, zorders):
    map = Basemap(llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2,
                  urcrnrlat=lat2, ax=ax)
    map.drawcoastlines(linewidth=3.0,
                       zorder=zorders['coastlines'])
    map.drawmapboundary(zorder=zorders['mapboundary'])
    # draw parallels and meridians.
    # label parallels on right and top
    # meridians on bottom and left
    parallels = np.arange(int(lat1), int(lat2), 2.)
    # labels = [left,right,top,bottom]
    map.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(int(lon1), int(lon2), 2.)
    map.drawmeridians(meridians,labels=[True,False,False,True])

def set_basemap_title(ax, tc_row, data_name):
    title_prefix = (f'IBTrACS wind radii and {data_name} ocean surface'
                    + f'wind speed of'
                    + f'\n{tc_row.sid}')
    if tc_row.name is not None:
        tc_name =  f'({tc_row.name}) '
    title_suffix = f'on {tc_row.date_time}'
    ax.set_title(f'{title_prefix} {tc_name} {title_suffix}')

"""
def draw_windspd(ax, lats, lons, windspd, zorders):
    # Plot windspd in knots with matplotlib's contour
    X, Y = np.meshgrid(lons, lats)
    Z = windspd

    windspd_levels = [5*x for x in range(15)]

    cs = ax.contour(X, Y, Z, levels=windspd_levels,
                    zorder=zorders['contour'], colors='k')
    ax.clabel(cs, inline=1, colors='k', fontsize=10)
    ax.contourf(X, Y, Z, levels=windspd_levels,
                zorder=zorders['contourf'],
                cmap=plt.cm.rainbow)
"""

def get_radii_from_tc_row(tc_row):
    r34 = dict()
    r34['nw'], r34['sw'], r34['se'], r34['ne'] = \
            tc_row.r34_nw, tc_row.r34_sw, tc_row.r34_se, tc_row.r34_ne

    r50 = dict()
    r50['nw'], r50['sw'], r50['se'], r50['ne'] = \
            tc_row.r50_nw, tc_row.r50_sw, tc_row.r50_se, tc_row.r50_ne

    r64 = dict()
    r64['nw'], r64['sw'], r64['se'], r64['ne'] = \
            tc_row.r64_nw, tc_row.r64_sw, tc_row.r64_se, tc_row.r64_ne

    radii = {34: r34, 50: r50, 64: r64}

    return radii

def draw_ibtracs_radii(ax, tc_row, zorders):
    center = get_tc_center(tc_row)
    tc_radii = get_radii_from_tc_row(tc_row)
    # radii_color = {34: 'yellow', 50: 'orange', 64: 'red'}
    radii_linestyle = {34: 'solid', 50: 'dashed', 64: 'dotted'}
    dirs = ['ne', 'se', 'sw', 'nw']
    ibtracs_area = []

    for r in RADII_LEVELS:
        area_in_radii = 0
        for idx, dir in enumerate(dirs):
            if tc_radii[r][dir] is None:
                continue

            ax.add_patch(
                mpatches.Wedge(
                    center,
                    r=tc_radii[r][dir]*DEGREE_OF_ONE_NMILE,
                    theta1=idx*90, theta2=(idx+1)*90,
                    zorder=zorders['wedge'],
                    #color=radii_color[r], alpha=0.6
                    fill=False, linestyle=radii_linestyle[r]
                )
            )

            radii_in_km = tc_radii[r][dir] * KM_OF_ONE_NMILE
            area_in_radii += math.pi * (radii_in_km)**2 / 4

        ibtracs_area.append(area_in_radii)

    return ibtracs_area

def get_area_within_radii(ax, lats, lons, windspd):
    X, Y = np.meshgrid(lons, lats)
    Z = windspd

    cs = ax.contour(X, Y, Z, levels=RADII_LEVELS)
    area = []
    for i in range(len(RADII_LEVELS)):
        if windspd.max() < RADII_LEVELS[i]:
            area.append(0)
            continue

        contour = cs.collections[i]
        paths = contour.get_paths()

        if not len(paths):
            continue

        vs = paths[0].vertices
        # Compute area enclosed by vertices.
        area.append(abs(
            area_of_contour(vs) * (KM_OF_ONE_DEGREE)**2))

    return area

def create_area_compare_table(the_class):
    """Get table of ERA5 reanalysis.

    """
    table_name = f'RADII_LEVELS_area_compare'

    class WindRadiiAreaCompare(object):
        pass

    if the_class.engine.dialect.has_table(the_class.engine, table_name):
        metadata = MetaData(bind=the_class.engine, reflect=True)
        t = metadata.tables[table_name]
        mapper(WindRadiiAreaCompare, t)

        return WindRadiiAreaCompare

    cols = []
    cols.append(Column('key', Integer, primary_key=True))
    cols.append(Column('sid', String(13), nullable=False))
    cols.append(Column('date_time', DateTime, nullable=False))
    for type in ['ibtracs', 'era5', 'smap']:
        for r in RADII_LEVELS:
            col_name = f'{type}_r{r}_area'
            cols.append(Column(col_name, Float, nullable=False))
    cols.append(Column('sid_date_time', String(50), nullable=False,
                       unique=True))

    metadata = MetaData(bind=the_class.engine)
    t = Table(table_name, metadata, *cols)
    mapper(WindRadiiAreaCompare, t)

    metadata.create_all()
    the_class.session.commit()

    return WindRadiiAreaCompare

def write_area_compare(the_class, tc_row, ibtracs_area,
                       area_type, area_to_compare):
    area = {
        'ibtracs': ibtracs_area,
        area_type: area_to_compare
    }

    CompareTable = create_area_compare_table(the_class)
    row = CompareTable()
    # Write area and metrics into row
    row.sid = tc_row.sid
    row.date_time = tc_row.date_time

    for type in ['ibtracs', area_type]:
        for idx, r in enumerate(RADII_LEVELS):
            setattr(row, f'{type}_r{r}_area', float(area[type][idx]))
    row.sid_date_time = f'{tc_row.sid}_{tc_row.date_time}'

    utils.bulk_insert_avoid_duplicate_unique(
        [row], the_class.CONFIG['database']\
        ['batch_size']['insert'],
        CompareTable, ['sid_date_time'], the_class.session,
        check_self=True)

def draw_compare_area_bar(ax, ibtracs_area, area_to_compare, data_name,
                          tc_row):
    labels = ['R34 area', 'R50 area', 'R64 area']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x - width/2, ibtracs_area, width,
                     label='IBTrACS')
    rects2 = ax.bar(x + width/2, area_to_compare, width,
                    label=data_name)

    set_bar_title_and_so_on(ax, tc_row, labels, x, data_name)

    utils.autolabel(ax, rects1)
    utils.autolabel(ax, rects2)

def set_bar_title_and_so_on(ax, tc_row, labels, x, data_name):
    title_prefix = (f'Area within wind radii of IBTrACS '
                    + f'and area within corresponding contour of '
                    + f'{data_name}\n of {tc_row.sid}')
    if tc_row.name is not None:
        tc_name =  f'({tc_row.name}) '
    title_suffix = f'on {tc_row.date_time}'

    ax.set_title(f'{title_prefix} {tc_name} {title_suffix}')
    ax.set_ylabel('Area')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

def get_basic_satel_columns():
    cols = []
    cols.append(Column('key', Integer, primary_key=True))
    cols.append(Column('datetime', DateTime, nullable=False))
    cols.append(Column('x', Integer, nullable=False))
    cols.append(Column('y', Integer, nullable=False))
    cols.append(Column('lon', Float, nullable=False))
    cols.append(Column('lat', Float, nullable=False))
    cols.append(Column('datetime_x_y', String(20), nullable=False,
                       unique=True))

    return cols

def get_basic_satel_era5_columns(tc_info=False):
    cols = []
    cols.append(Column('key', Integer, primary_key=True))
    if tc_info:
        cols.append(Column('sid', String(13), nullable=False))
    cols.append(Column('satel_datetime', DateTime, nullable=False))
    cols.append(Column('era5_datetime', DateTime, nullable=False))
    cols.append(Column('x', Integer, nullable=False))
    cols.append(Column('y', Integer, nullable=False))
    cols.append(Column('lon', Float, nullable=False))
    cols.append(Column('lat', Float, nullable=False))
    cols.append(Column('satel_datetime_lon_lat', String(50),
                       nullable=False, unique=True))

    return cols

class GetChildDataIndices(object):
    def __init__(self, grid_pts, lats, lons, owi_az_size):
        self.grid_pts = grid_pts
        self.lats = lats
        self.lons = lons
        self.owi_az_size = owi_az_size

def get_data_indices_around_grid_pts(workload):
    grid_pts = workload.grid_pts
    lats = workload.lats
    lons = workload.lons
    owi_az_size = workload.owi_az_size

    total = len(grid_pts) * owi_az_size
    count = 0
    pt_region_data_indices = []

    for pt_idx, pt in enumerate(grid_pts):
        pt_region_data_indices.append(list())

        for i in range(owi_az_size):
            count += 1
            percent = float(count) / total * 100
            print((f"""\rTask ({os.getpid()} """
                   f"""is getting data indices around ocean """
                   f"""grid points {percent:.2f}%"""),
                  end = '')

            lat_row, lon_row = lats[i], lons[i]
            lat_match_indices = [i for i,v in enumerate(
                abs(lat_row - pt.lat) < 0.025) if v]
            lon_match_indices = [i for i,v in enumerate(
                abs(lon_row - pt.lon) < 0.025) if v]

            match_indices = np.intersect1d(lat_match_indices,
                                           lon_match_indices)
            for j in match_indices:
                pt_region_data_indices[pt_idx].append((i, j))

    delete_last_lines()
    print('Done')

    return pt_region_data_indices

def gen_satel_era5_tablename(satel_name, dt):
    return f'{satel_name}_{dt.year}_{str(dt.month).zfill(2)}'

def gen_tc_satel_era5_tablename(satel_name, dt, basin):
    if isinstance(dt, datetime.datetime):
        return f'tc_{satel_name}_era5_{dt.year}_{basin}'
    # `dt` is year
    elif isinstance(dt, int):
        return f'tc_{satel_name}_era5_{dt}_{basin}'

def backtime_to_last_entire_hour(dt):
    # Initial datetime is not entire-houred
    if dt.minute or dt.second or dt.microsecond:
        dt = dt.replace(second=0, microsecond=0, minute=0,
                        hour=dt.hour)

    return dt

def forwardtime_to_next_entire_hour(dt):
    # Initial datetime is not entire-houred
    dt = (dt.replace(second=0, microsecond=0, minute=0,
                     hour=dt.hour)
          + datetime.timedelta(hours=1))

    return dt

def load_grid_lonlat_xy(the_class):
    grid_pickles = the_class.CONFIG['grid']['pickle']

    for key in grid_pickles.keys():
        with open(grid_pickles[key], 'rb') as f:
            var = pickle.load(f)

        setattr(the_class, f'grid_{key}', var)

def decompose_wind(windspd, winddir, input_convention):
    """Decompose windspd with winddir into u and v component of wind.

    Parameters
    ----------
    windspd: float
        Wind speed.
    winddir: float
        Wind direction in degree.  It increases clockwise from North
        when viewed from above.
    input_convention: str
        Convention of inputted wind direction.  'o' means oceanographic
        convention and 'm' means meteorological convention.

    Returns
    -------
    u_wind: float
        U component of wind.  None if input is invalid.
    v_wind: float
        V component of wind.  None if input is invalid.

    """
    if windspd is None or winddir is None:
        return None, None

    # Oceanographic convention
    if input_convention == 'o':
        u_wind = windspd * math.sin(math.radians(winddir))
        v_wind = windspd * math.cos(math.radians(winddir))
    # Meteorological convention
    elif input_convention == 'm':
        u_wind = -windspd * math.sin(math.radians(winddir))
        v_wind = -windspd * math.cos(math.radians(winddir))

    return u_wind, v_wind

def compose_wind(u_wind, v_wind, output_convention):
    """Compose windspd and winddir from u and v component of wind.

    Parameters
    ----------
    u_wind: float
        U component of wind.
    v_wind: float
        V component of wind.
    output_convention: str
        Convention of output wind direction.  'o' means oceanographic
        convention and 'm' means meteorological convention.

    Returns
    -------
    windspd: float
        Wind speed.
    winddir: float
        Wind direction in degree.  It increases clockwise from North
        when viewed from above.

    """
    windspd = math.sqrt(u_wind ** 2 + v_wind ** 2)

    # Oceanographic convention
    if output_convention == 'o':
        winddir = math.degrees(math.atan2(u_wind, v_wind))
    # Meteorological convention
    elif output_convention == 'm':
        winddir = math.degrees(math.atan2(-u_wind, -v_wind))

    winddir = (winddir + 360) % 360

    return windspd, winddir

def get_dataframe_cols_with_no_nans(df, col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)

    return cols_with_no_nans

def gen_scs_era5_table_name(dt_cursor, hourtime):
    table_name = (f"""era5_scs_{dt_cursor.strftime('%Y_%m%d')}"""
                  f"""_{str(hourtime).zfill(4)}""")

    return table_name

def draw_windspd_with_contourf(fig, ax, lons, lats, windspd,
                               wind_zorder, max_windspd, mesh):
    if not mesh:
        X, Y = np.meshgrid(lons, lats)
    else:
        X, Y = lons, lats
    Z = windspd

    # windspd_levels = [5*x for x in range(1, 15)]
    windspd_levels = np.linspace(0, max_windspd, 30)

    # cs = ax.contour(X, Y, Z, levels=windspd_levels,
    #                 zorder=self.zorders['contour'], colors='k')
    # ax.clabel(cs, inline=1, colors='k', fontsize=10)
    try:
        cf = ax.contourf(X, Y, Z,
                         levels=windspd_levels,
                         zorder=wind_zorder,
                         cmap=plt.cm.rainbow,
                         vmin=0, vmax=max_windspd)
    except Exception as msg:
        breakpoint()
        exit(msg)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    clb = fig.colorbar(cf, cax=cax, orientation='vertical',
                       format='%.1f')
    clb.ax.set_title('m/s')

def draw_SCS_basemap(the_class, ax, custom, region):
    if not custom:
        lat1 = the_class.lat1
        lat2 = the_class.lat2
        lon1 = the_class.lon1
        lon2 = the_class.lon2
    else:
        # South, North, West, East
        lat1, lat2, lon1, lon2 = region

    adjust = 0.0
    map = Basemap(llcrnrlon=lon1-adjust, llcrnrlat=lat1-adjust,
                  urcrnrlon=lon2+adjust, urcrnrlat=lat2+adjust,
                  ax=ax, resolution='h')

    map.drawcoastlines(zorder=the_class.zorders['coastlines'])
    map.drawmapboundary(fill_color='white', linewidth=1,
                        zorder=the_class.zorders['mapboundary'])
    map.fillcontinents(color='grey', lake_color='white',
                       zorder=the_class.zorders['continents'])

    meridians_interval = (lon2 - lon1) / 4
    parallels_interval = (lat2 - lat1) / 4
    map.drawmeridians(np.arange(lon1, lon2+0.01, meridians_interval),
                      labels=[1, 0, 0, 1],  fmt='%.2f',
                      zorder=the_class.zorders['grid'])
    map.drawparallels(np.arange(lat1, lat2+0.01, parallels_interval),
                      labels=[1, 0 , 0, 1], fmt='%.2f',
                      zorder=the_class.zorders['grid'])

    return map

def draw_windspd_with_imshow(map, fig, ax, lons, lats, windspd,
                               wind_zorder, max_windspd, mesh):
    rows_num, cols_num = windspd.shape
    for i in range(rows_num):
        for j in range(cols_num):
            if windspd[i][j] < 0:
                windspd[i][j] = None

    if not mesh:
        X, Y = np.meshgrid(lons, lats)
    else:
        X, Y = lons, lats
    Z = windspd

    try:
        image = map.imshow(Z, ax=ax, zorder=wind_zorder,
                           cmap=plt.cm.rainbow,
                           vmin=0, vmax=max_windspd,
                           interpolation='none',
                           extent=(X.min(), X.max(), Y.min(), Y.max()))
        # plt.colorbar()
    except Exception as msg:
        breakpoint()
        exit(msg)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    clb = fig.colorbar(image, cax=cax, orientation='vertical',
                       format='%.1f')
    clb.ax.set_title('m/s')

def draw_windspd(the_class, fig, ax, dt, lons, lats, windspd,
                 max_windspd, mesh, custom=False, region=None):
    map = draw_SCS_basemap(the_class, ax, custom, region)

    draw_windspd_with_contourf(fig, ax, lons, lats, windspd,
                               the_class.zorders['contourf'],
                               max_windspd, mesh)

    # draw_windspd_with_imshow(map, fig, ax, lons, lats, windspd,
    #                          the_class.zorders['contourf'],
    #                            max_windspd, mesh)

def get_latlon_and_index_in_grid(value, range, grid_lat_or_lon_list):
    """Getting region corners' lat or lon's value and its index
    in RSS grid.

    Parameters
    ----------
    value: float
        The value of latitude or longtitude of region corner.
    range: tuple
        The range of latitude or longtitude of region.  The first
        element is smaller than the second element.
    grid_lat_or_lon_list: list
        The latitude or longitutde of RSS grid.  Ascending sorted.

    Return
    ------
    grid_pt_value: float
        The value of latitude or longtitude of matching RSS grid point.
    grid_pt_index: int
        The index of latitude or longtitude of matching RSS grid point.

    """
    tmp_value, tmp_index = get_nearest_element_and_index(
        grid_lat_or_lon_list, value)

    # To avoid gap near margin of map due to little difference
    # between different grid, expand region a little

    # value is the first element (smaller one) of range
    if not range.index(value):
        if tmp_value > value:
            tmp_value -= 0.25
            tmp_index -= 1
    # value is the second element (larger one) of range
    else:
        if tmp_value < value:
            tmp_value += 0.25
            tmp_index += 1

    grid_pt_value = tmp_value
    grid_pt_index = tmp_index

    return grid_pt_value, grid_pt_index

def get_nearest_element_and_index(list_, element):
    ae = [abs(element - x) for x in list_]
    match_index = ae.index(min(ae))

    return list_[match_index], match_index

def get_pixel_of_smap_windspd(smap_file_path, dt, lon, lat):
    spa_resolu = 0.25

    smap_lats = [y * spa_resolu - 89.875 for y in range(720)]
    smap_lons = [x * spa_resolu + 0.125 for x in range(1440)]

    match_lat, lat_match_index = get_nearest_element_and_index(
        smap_lats, lat)
    match_lon, lon_match_index = get_nearest_element_and_index(
        smap_lons, lon)

    dataset = netCDF4.Dataset(smap_file_path)
    # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which
    # faster than 1 m/s, so must disable auto mask
    dataset.set_auto_mask(False)
    vars = dataset.variables
    minute = vars['minute']
    wind = vars['wind']

    y = lat_match_index
    x = lon_match_index
    windspd = None

    passes_num = 2
    minute_missing = -9999
    wind_missing = -99.99
    for i in range(passes_num):
        if (minute[y][x][i] == minute_missing
            or wind[y][x][i] == wind_missing):
            continue
        if minute[y][x][0] == minute[y][x][1]:
            break
        time_ = datetime.time(
            *divmod(int(minute[y][x][i]), 60), 0)
        # Temporal window is one hour
        if time_.hour != dt.hour - 1 and time_.hour != dt.hour:
            continue

        # SMAP originally has land mask, so it's not necessary
        # to check whether each pixel is land or ocean
        windspd = float(wind[y][x][i])
        break

    return windspd

def get_xyz_matrix_of_smap_windspd_or_diff_mins(
    target, smap_file_path, tc_dt, region):
    """Temporal window is one hour.

    """
    spa_resolu = 0.25
    windspd_masked_value = -999
    diff_mins_masked_value = -999

    smap_lats = [y * spa_resolu - 89.875 for y in range(720)]
    smap_lons = [x * spa_resolu + 0.125 for x in range(1440)]

    lat1, lat1_idx = get_latlon_and_index_in_grid(
        region[0], (region[0], region[1]), smap_lats)
    lat2, lat2_idx = get_latlon_and_index_in_grid(
        region[1], (region[0], region[1]), smap_lats)
    lon1, lon1_idx = get_latlon_and_index_in_grid(
        region[2], (region[2], region[3]), smap_lons)
    lon2, lon2_idx = get_latlon_and_index_in_grid(
        region[3], (region[2], region[3]), smap_lons)

    lons = list(np.arange(lon1, lon2 + 0.5 * spa_resolu, spa_resolu))
    lats = list(np.arange(lat1, lat2 + 0.5 * spa_resolu, spa_resolu))
    lons = [round(x, 3) for x in lons]
    lats = [round(y, 3) for y in lats]
    windspd = np.full(shape=(len(lats), len(lons)),
                      fill_value=windspd_masked_value, dtype=float)
    diff_mins = np.full(shape=(len(lats), len(lons)),
                        fill_value=diff_mins_masked_value, dtype=int)

    dataset = netCDF4.Dataset(smap_file_path)
    # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which
    # faster than 1 m/s, so must disable auto mask
    dataset.set_auto_mask(False)
    vars = dataset.variables
    minute = vars['minute'][lat1_idx:lat2_idx+1, lon1_idx:lon2_idx+1, :]
    wind = vars['wind'][lat1_idx:lat2_idx+1, lon1_idx:lon2_idx+1, :]

    rows, cols = minute.shape[:2]
    passes_num = 2
    minute_missing = -9999
    wind_missing = -99.99
    for y in range(len(lats)):
        for x in range(len(lons)):
            for i in range(passes_num):
                try:
                    if (minute[y][x][i] == minute_missing
                        or wind[y][x][i] == wind_missing):
                        continue
                    if minute[y][x][0] == minute[y][x][1]:
                        continue
                    if minute[y][x][i] == 1440:
                        continue
                    pt_time = datetime.time(
                        *divmod(int(minute[y][x][i]), 60), 0)
                    pt_dt = datetime.datetime.combine(tc_dt.date(),
                                                      pt_time)
                    delta = abs(pt_dt - tc_dt)
                    # Temporal window is one hour
                    if delta.seconds > 1800:
                        continue

                    # SMAP originally has land mask, so it's not
                    # necessary to check whether each pixel is land
                    # or ocean
                    windspd[y][x] = float(wind[y][x][i])
                    if pt_dt < tc_dt:
                        diff_mins[y][x] = -int(delta.seconds / 60)
                    else:
                        diff_mins[y][x] = int(delta.seconds / 60)
                except Exception as msg:
                    breakpoint()
                    exit(msg)

    if target == 'windspd' and windspd.max() > windspd_masked_value:
        return lons, lats, windspd
    elif (target == 'diff_mins'
          and diff_mins.max() > diff_mins_masked_value):
        return lons, lats, diff_mins
    else:
        return None, None, None

# def get_xyz_matrix_of_smap_windspd(smap_file_path, tc_dt, region):
#     """Temporal window is one hour.
# 
#     """
#     spa_resolu = 0.25
# 
#     smap_lats = [y * spa_resolu - 89.875 for y in range(720)]
#     smap_lons = [x * spa_resolu + 0.125 for x in range(1440)]
# 
#     lat1, lat1_idx = get_latlon_and_index_in_grid(
#         region[0], (region[0], region[1]), smap_lats)
#     lat2, lat2_idx = get_latlon_and_index_in_grid(
#         region[1], (region[0], region[1]), smap_lats)
#     lon1, lon1_idx = get_latlon_and_index_in_grid(
#         region[2], (region[2], region[3]), smap_lons)
#     lon2, lon2_idx = get_latlon_and_index_in_grid(
#         region[3], (region[2], region[3]), smap_lons)
# 
#     lons = list(np.arange(lon1, lon2 + 0.5 * spa_resolu, spa_resolu))
#     lats = list(np.arange(lat1, lat2 + 0.5 * spa_resolu, spa_resolu))
#     lons = [round(x, 3) for x in lons]
#     lats = [round(y, 3) for y in lats]
#     windspd = np.full(shape=(len(lats), len(lons)), fill_value=-1,
#                       dtype=float)
# 
#     dataset = netCDF4.Dataset(smap_file_path)
#     # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which
#     # faster than 1 m/s, so must disable auto mask
#     dataset.set_auto_mask(False)
#     vars = dataset.variables
#     minute = vars['minute'][lat1_idx:lat2_idx+1, lon1_idx:lon2_idx+1, :]
#     wind = vars['wind'][lat1_idx:lat2_idx+1, lon1_idx:lon2_idx+1, :]
# 
#     rows, cols = minute.shape[:2]
#     passes_num = 2
#     minute_missing = -9999
#     wind_missing = -99.99
#     for y in range(len(lats)):
#         for x in range(len(lons)):
#             for i in range(passes_num):
#                 try:
#                     if (minute[y][x][i] == minute_missing
#                         or wind[y][x][i] == wind_missing):
#                         continue
#                     if minute[y][x][0] == minute[y][x][1]:
#                         continue
#                     pt_time = datetime.time(
#                         *divmod(int(minute[y][x][i]), 60), 0)
#                     pt_dt = datetime.datetime.combine(tc_dt.date(), pt_time)
#                     delta = abs(pt_dt - tc_dt)
#                     # Temporal window is one hour
#                     if delta.seconds > 1800:
#                         continue
#                         continue
# 
#                     # SMAP originally has land mask, so it's not necessary
#                     # to check whether each pixel is land or ocean
#                     windspd[y][x] = float(wind[y][x][i])
#                 except Exception as msg:
#                     breakpoint()
#                     exit(msg)
# 
#     if windspd.max() > 0:
#         return lons, lats, windspd
#     else:
#         return None, None, None

def satel_data_cover_tc_center(lons, lats, windspd, tc):
    # (lon, lat)
    tc_lon, tc_lat = get_tc_center(tc)
    tc_lon_in_grid, tc_lon_in_grid_idx = \
            get_nearest_element_and_index(lons, tc_lon)
    tc_lat_in_grid, tc_lat_in_grid_idx = \
            get_nearest_element_and_index(lats, tc_lat)

    if windspd[tc_lat_in_grid_idx][tc_lon_in_grid_idx] > 0:
        return True
    else:
        return False

def cover_tc_wind_radii(lons, lats, windspd, tc):
    # (lon, lat)
    center = get_tc_center(tc_row)
    tc_radii = get_radii_from_tc_row(tc_row)

    largest_radii = RADII_LEVELS[0]

    pass

def draw_ibtracs_radii(ax, tc_row, zorders):
    center = get_tc_center(tc_row)
    tc_radii = get_radii_from_tc_row(tc_row)
    # radii_color = {34: 'yellow', 50: 'orange', 64: 'red'}
    radii_linestyle = {34: 'solid', 50: 'dashed', 64: 'dotted'}
    dirs = ['ne', 'se', 'sw', 'nw']
    ibtracs_area = []

    for r in RADII_LEVELS:
        area_in_radii = 0
        for idx, dir in enumerate(dirs):
            if tc_radii[r][dir] is None:
                continue

            ax.add_patch(
                mpatches.Wedge(
                    center,
                    r=tc_radii[r][dir]*DEGREE_OF_ONE_NMILE,
                    theta1=idx*90, theta2=(idx+1)*90,
                    zorder=zorders['wedge'],
                    #color=radii_color[r], alpha=0.6
                    fill=False, linestyle=radii_linestyle[r]
                )
            )

            radii_in_km = tc_radii[r][dir] * KM_OF_ONE_NMILE
            area_in_radii += math.pi * (radii_in_km)**2 / 4

        ibtracs_area.append(area_in_radii)

    return ibtracs_area

def get_xyz_matrix_of_ccmp_windspd(ccmp_file_path, dt, region):
    ccmp_hours = [0, 6, 12, 18]
    if dt.hour not in ccmp_hours:
        return None, None, None
    else:
        hour_idx = ccmp_hours.index(dt.hour)

    spa_resolu = 0.25

    ccmp_lats = [y * spa_resolu - 78.375 for y in range(628)]
    ccmp_lons = [x * spa_resolu + 0.125 for x in range(1440)]

    lat1, lat1_idx = get_latlon_and_index_in_grid(
        region[0], (region[0], region[1]), ccmp_lats)
    lat2, lat2_idx = get_latlon_and_index_in_grid(
        region[1], (region[0], region[1]), ccmp_lats)
    lon1, lon1_idx = get_latlon_and_index_in_grid(
        region[2], (region[2], region[3]), ccmp_lons)
    lon2, lon2_idx = get_latlon_and_index_in_grid(
        region[3], (region[2], region[3]), ccmp_lons)

    lons = list(np.arange(lon1, lon2 + 0.5 * spa_resolu, spa_resolu))
    lats = list(np.arange(lat1, lat2 + 0.5 * spa_resolu, spa_resolu))
    lons = [round(x, 3) for x in lons]
    lats = [round(y, 3) for y in lats]
    windspd = np.ndarray(shape=(len(lats), len(lons)), dtype=float)

    vars = netCDF4.Dataset(ccmp_file_path).variables
    u_wind = vars['uwnd'][hour_idx]\
            [lat1_idx:lat2_idx+1, lon1_idx:lon2_idx+1]
    v_wind = vars['vwnd'][hour_idx]\
            [lat1_idx:lat2_idx+1, lon1_idx:lon2_idx+1]

    for y in range(len(lats)):
        for x in range(len(lons)):
            try:
                if not bool(globe.is_land(lats[y], lons[x])):
                    # For safe, maybe should check whether wind
                    # component is masked
                    windspd[y][x] = math.sqrt(u_wind[y][x] ** 2
                                              + v_wind[y][x] ** 2)
                else:
                    # There may be problem
                    windspd[y][x] = 0
            except Exception as msg:
                breakpoint()
                exit(msg)

    return lons, lats, windspd

def get_pixel_of_era5_windspd(era5_file_path, product_type, dt,
                              lon, lat):
    grbidx = pygrib.index(era5_file_path, 'dataTime')
    hourtime = dt.hour * 100
    selected_grbs = grbidx.select(dataTime=hourtime)

    if product_type == 'single_levels':
        u_wind_var_name = 'neutral_wind_at_10_m_u-component'
        v_wind_var_name = 'neutral_wind_at_10_m_v-component'
    elif product_type == 'pressure_levels':
        u_wind_var_name = 'u_component_of_wind'
        v_wind_var_name = 'v_component_of_wind'

    spa_resolu = 0.25
    era5_lats = [y * spa_resolu - 90 for y in range(721)]
    era5_lons = [x * spa_resolu for x in range(1440)]
    match_lat, lat_match_index = get_nearest_element_and_index(
        era5_lats, lat)
    match_lon, lon_match_index = get_nearest_element_and_index(
        era5_lons, lon)
    u_wind = None
    v_wind = None

    for grb in selected_grbs:
        try:
            name = grb.name.replace(" ", "_").lower()

            data, lats, lons = grb.data(-90, 90, 0, 360)
            data = np.flip(data, 0)

            if name == u_wind_var_name:
                u_wind = data[lat_match_index][lon_match_index]
            elif name == v_wind_var_name:
                v_wind = data[lat_match_index][lon_match_index]
        except Exception as msg:
            breakpoint()
            exit(msg)

    try:
        windspd = math.sqrt(u_wind ** 2 + v_wind ** 2)
    except Exception as msg:
        breakpoint()
        exit(msg)

    return windspd

def get_xyz_matrix_of_era5_windspd(era5_file_path, product_type, dt, region):
    grbidx = pygrib.index(era5_file_path, 'dataTime')
    hourtime = dt.hour * 100
    selected_grbs = grbidx.select(dataTime=hourtime)

    u_wind = None
    v_wind = None

    if product_type == 'single_levels':
        u_wind_var_name = 'neutral_wind_at_10_m_u-component'
        v_wind_var_name = 'neutral_wind_at_10_m_v-component'
    elif product_type == 'pressure_levels':
        u_wind_var_name = 'u_component_of_wind'
        v_wind_var_name = 'v_component_of_wind'

    for grb in selected_grbs:
        name = grb.name.replace(" ", "_").lower()

        data, lats, lons = grb.data(region[0], region[1],
                                    region[2], region[3])
        data = np.flip(data, 0)
        lats = np.flip(lats, 0)
        lons = np.flip(lons, 0)

        if name == u_wind_var_name:
            u_wind = data
        elif name == v_wind_var_name:
            v_wind = data

    lats_num, lons_num = u_wind.shape
    windspd = np.ndarray(shape=u_wind.shape, dtype=float)
    for y in range(lats_num):
        for x in range(lons_num):
            try:
                lon_180_mode = longtitude_converter(lons[y][x], '360',
                                                    '-180')
                if not bool(globe.is_land(lats[y][x], lon_180_mode)):
                    windspd[y][x] = math.sqrt(u_wind[y][x] ** 2
                                              + v_wind[y][x] ** 2)
                else:
                    # There may be problem
                    windspd[y][x] = 0
            except Exception as msg:
                breakpoint()
                exit(msg)

    return lons, lats, windspd

def if_mesh(lons):
    if isinstance(lons, list):
        return False
    elif isinstance(lons, np.ndarray):
        if len(lons.shape) == 2:
            return True
        return True

def get_subplots_row_col_and_fig_size(subplots_num):
    if subplots_num == 1:
        return 1, 1, (7, 7)
    elif subplots_num == 2:
        return 1, 2, (15, 7)
    elif subplots_num == 3 or subplots_num == 4:
        return 2, 2, (15, 15)
    else:
        logger.error('Too many subplots, should not more than 4.')
        exit()

def get_era5_corners_of_rss_cell(lat, lon, era5_lats_grid,
                                 era5_lons_grid, grb_spa_resolu):
    era5_lats = list(era5_lats_grid[:, 0])
    era5_lons = list(era5_lons_grid[0, :])

    try:
        if grb_spa_resolu == 0.25:
            delta = 0.5 * 0.25
            lat1 = lat - delta
            lat2 = lat + delta
            lon1 = lon - delta
            lon2 = (lon + delta) % 360

            lat1_idx = era5_lats.index(lat1)
            lat2_idx = era5_lats.index(lat2)
            lon1_idx = era5_lons.index(lon1)
            lon2_idx = era5_lons.index(lon2)
        elif grb_spa_resolu == 0.5:
            nearest_lat, nearest_lat_idx = get_nearest_element_and_index(
                era5_lats, lat)
            if nearest_lat < lat:
                lat1 = nearest_lat
                lat1_idx = nearest_lat_idx
                lat2_idx = lat1_idx + 1
                lat2 = era5_lats[lat2_idx]
            else:
                lat2 = nearest_lat
                lat2_idx = nearest_lat_idx
                lat1_idx = lat2_idx - 1
                lat1 = era5_lats[lat1_idx]

            nearest_lon, nearest_lon_idx = get_nearest_element_and_index(
                era5_lons, lon)
            if nearest_lon < lon:
                lon1 = nearest_lon
                lon1_idx = nearest_lon_idx
                lon2_idx = lon1_idx + 1
                lon2 = era5_lons[lon2_idx]
            else:
                lon2 = nearest_lon
                lon2_idx = nearest_lon_idx
                lon1_idx = lon2_idx - 1
                lon1 = era5_lons[lon1_idx]
    except Exception as msg:
        breakpoint()
        # exit(msg)

    return [lat1, lat2, lon1, lon2], [lat1_idx, lat2_idx, lon1_idx,
                                      lon2_idx]

def find_neighbours_of_pt_in_half_degree_grid(pt):
    nearest = round(pt * 2) / 2
    direction = ((pt - nearest) / abs(pt - nearest))
    the_other = pt + (0.5 - abs(pt- nearest)) * direction

    return min(nearest, the_other), max(nearest, the_other)

def get_center_shift_of_two_tcs(next_tc, tc):
    next_tc_center = get_tc_center(next_tc)
    tc_center = get_tc_center(tc)
    # Set the value to check whether the line between two TC center across
    # the prime meridian
    threshold = 20
    if abs(tc_center[0] - next_tc_center[0]) > threshold:
        if tc_center[0] < next_tc_center[0]:
            # E.g. `tc` lon: 0.5, `next_tc` lon: 359.5
            tc_center = (tc_center[0] + 360, tc_center[1])
        elif tc_center[0] > next_tc_center[0]:
            # E.g. `tc` lon: 359.5, `next_tc` lon: 0.5
            next_tc_center = (next_tc_center[0] + 360, next_tc_center[1])

    lons_shift = next_tc_center[0] - tc_center[0]
    lats_shift = next_tc_center[1] - tc_center[1]

    return lons_shift, lats_shift

def get_tc_center(tc_row):
    lon_converted = tc_row.lon + 360 if tc_row.lon < 0 else tc_row.lon
    center = (lon_converted, tc_row.lat)

    return center

def process_grib_message_name(name):
    return name.replace(" ", "_").replace("-", "_").replace('(', '')\
            .replace(')', '').lower()

def longtitude_converter(lon, input_mode, output_mode):
    if input_mode == '360' and output_mode == '-180':
        if lon > 180:
            lon -= 360
    elif input_mode == '-180' and output_mode == '360':
        if lon < 0:
            lon += 360

    return lon

def find_best_NN_weights_file(dir):
    file_names = [f for f in os.listdir(dir) if f.endswith('.hdf5')]
    max_epoch = -1
    best_weights_file_name = None

    for file in file_names:
        epoch = int(file.split('-')[1])
        if epoch > max_epoch:
            max_epoch = epoch
            best_weights_file_name = file

    return f'{dir}{best_weights_file_name}'

def filter_dataframe_by_column_value_divide(df, col_name, divide,
                                            large_or_small):
    if large_or_small == 'large':
        in_range = df[col_name] > divide
    elif large_or_small == 'small':
        in_range = df[col_name] < divide

    condition = f'{col_name}_{large_or_small}_than_{divide}'

    return df[in_range], condition

def is_multiple_of(a, b):
    result = a % b
    return (result < 1e-3)

def get_sharpened_lats_of_era5_ocean_grid(
    lats, new_lats_num, new_lons_num, spa_resolu_diff):
    """Increase resolution of new lats grid to half of before.

    """
    new_lats = np.ndarray(shape=(new_lats_num, new_lons_num),
                          dtype=float)
    lats_num = lats.shape[0]

    for y in range(lats_num - 1):
        lat = lats[y][0]
        for x in range(new_lons_num):
            new_lats[y * 2][x] = lat
        lat += spa_resolu_diff
        for x in range(new_lons_num):
            new_lats[y * 2 + 1][x] = lat
    lat = lats[-1][0]
    for x in range(new_lons_num):
        new_lats[-1][x] = lat

    return new_lats

def get_sharpened_lons_of_era5_ocean_grid(
    lons, new_lats_num, new_lons_num, spa_resolu_diff):
    """Increase resolution of new lons grid to half of before.

    """
    new_lons = np.ndarray(shape=(new_lats_num, new_lons_num),
                          dtype=float)
    lons_num = lons.shape[1]

    for x in range(lons_num - 1):
        lon = lons[0][x]
        for y in range(new_lats_num):
            new_lons[y][x * 2] = lon
        lon += spa_resolu_diff
        for y in range(new_lats_num):
            new_lons[y][x * 2 + 1] = lon
    lon = lons[0][-1]
    for y in range(new_lats_num):
        new_lons[y][-1] = lon

    return new_lons

def sharpen_era5_ocean_grid(data, lats, lons):
    """Sharpen ERA5 ocean grid to resolution of 0.25 x 0.25 degree.
    If `data` is an instance of numpy.ma.core.MaskedArray, the returned
    `new_data` is also an instance of numpy.ma.core.MaskedArray.

    """
    spa_resolu_diff = 0.25
    lats_num = lats.shape[0]
    lons_num = lons.shape[1]
    new_lats_num = lats_num * 2 - 1
    new_lons_num = lons_num * 2 - 1

    new_lats = get_sharpened_lats_of_era5_ocean_grid(
        lats, new_lats_num, new_lons_num, spa_resolu_diff)
    new_lons = get_sharpened_lons_of_era5_ocean_grid(
        lons, new_lats_num, new_lons_num, spa_resolu_diff)
    masked_value = -999
    new_data = np.full(shape=(new_lats_num, new_lons_num),
                       fill_value=masked_value, dtype=float)

    if isinstance(data, np.ma.core.MaskedArray):
        masked = True
    else:
        masked = False

    new_data = fill_era5_masked_and_not_masked_ocean_data(
        masked, data, new_data, lats, lons, masked_value)

    return new_data, new_lats, new_lons

def fill_era5_masked_and_not_masked_ocean_data(
    masked, data, new_data, lats, lons, masked_value):
    """

    """
    lats_num = data.shape[0]
    lons_num = data.shape[1]
    new_lats_num = lats_num * 2 - 1
    new_lons_num = lons_num * 2 - 1

    # Project from data to new_data
    if masked:
        for y in range(lats_num):
            for x in range(lons_num):
                if not data.mask[y][x]:
                    new_data[2 * y][2 * x] = data[y][x]
    else:
        for y in range(lats_num):
            for x in range(lons_num):
                new_data[2 * y][2 * x] = data[y][x]

    # Fill gaps if possible
    for y in range(lats_num):
        if y + 1 >= lats_num:
            break
        for x in range(lons_num):
            if x + 1 >= lons_num:
                break
            if masked:
                # Check if there is a minimum square with four corners
                whole_square = True
                for tmp_y in [y, y + 1]:
                    for tmp_x in [x, x + 1]:
                        if data.mask[tmp_y][tmp_x]:
                            whole_square = False
                            break
                    if not whole_square:
                        break
            # Interpolate in square
            #
            # Before
            # ------
            # fill   masked fill
            # masked masked masked
            # fill   masked fill
            #
            # After
            # -----
            # fill   fill   fill
            # fill   fill   fill
            # fill   fill   fill
            if not masked or (masked and whole_square):
                square_data = data[y:y+2, x:x+2]
                square_lats = lats[y:y+2, x:x+2]
                square_lons = lons[y:y+2, x:x+2]
                f = interpolate.interp2d(square_lons, square_lats,
                                         square_data)
                # center
                center_lon = (lons[y][x] + lons[y][x + 1]) / 2
                center_lat = (lats[y][x] + lats[y + 1][x]) / 2
                value = f(center_lon, center_lat)
                new_data[2 * y + 1][2 * x + 1] = value
                # top
                top_lon = center_lon
                top_lat = lats[y][x]
                value = f(top_lon, top_lat)
                new_data[2 * y][2 * x + 1] = value
                # bottom
                bottom_lon = center_lon
                bottom_lat = lats[y + 1][x]
                value = f(bottom_lon, bottom_lat)
                new_data[2 * y + 2][2 * x + 1] = value
                # left
                left_lon = lons[y][x]
                left_lat = center_lat
                value = f(left_lon, left_lat)
                new_data[2 * y + 1][2 * x] = value
                # right
                right_lon = lons[y][x + 1]
                right_lat = center_lat
                value = f(right_lon, right_lat)
                new_data[2 * y + 1][2 * x + 2] = value

    if masked:
        # Make new_data to be a masked array
        new_data = np.ma.masked_where(new_data==masked_value, new_data)

    return new_data

def load_best_xgb_model(model_dir, basin):
    model_files = [f for f in os.listdir(model_dir)
                   if (f.startswith(f'{basin}')
                       and f.endswith('.pickle.dat'))]
    min_mse = 99999999
    best_model_name = None
    # Find best model
    for file in model_files:
        mse = float(file.split('.pickle')[0].split('mse_')[1])
        if mse < min_mse:
            min_mse = mse
            best_model_name = file
    # load model from file
    best_model = pickle.load(
        open(f'{model_dir}{best_model_name}', 'rb'))

    return best_model

def sfmr_nc_converter(var_name, value, file_path=None):
    value = str(value)
    try:
        if var_name == 'DATE':
            return datetime.datetime.strptime(value,
                                              '%Y%m%d').date()
        elif var_name == 'TIME':
            return sfmr_nc_time_converter(value)
    except Exception as msg:
        breakpoint()
        exit(msg)

def sfmr_vars_filter(var_name, nc_var, masked_value,
                     time_masked_indices=None):
    # NetCDF original variable is read-only
    # So make a copy of it
    var = []
    for val in nc_var:
        var.append(val)

    result = []
    masked_indices = []
    invalid_value = 0

    # Although for TIME, there are two possibilities:
    # One: True invalid value
    # Two: 00:00:00 is invalid value 0

    # However, in SFMR files, there are TWO possibilities when TIME is
    # invalid value 0:
    # One: For 00:00:00, corresponding variables (DATE, LAT, LON)
    # are 0.
    # Two: For 00:00:00, corresponding variables (DATE, LAT, LON)
    # are valid values.
    if var_name == 'TIME':
        for idx, val in enumerate(var):
            if val == invalid_value:
                # Not around terminals
                if idx >= 1 and idx < len(var) - 1:
                    # So if any side of the point with invalid value
                    # is invalid value too, then this point is
                    # invalid
                    if (var[idx - 1] == invalid_value
                        or var[idx + 1] == invalid_value):
                        var[idx] = masked_value
                else:
                    var[idx] = masked_value
    elif var_name == 'DATE' or var_name == 'LON' or var_name == 'LAT':
        # Assuming that SFMR data are all in NA around USA,
        # so the longitude and latitude are far away from 0
        for idx, val in enumerate(var):
            if val == invalid_value:
                if idx in time_masked_indices:
                    var[idx] = masked_value
                # When TIME is 00:00:00, given invalid DATE/LON/LAT
                # a neighbouring value
                else:
                    if idx >= 1:
                        var[idx] = var[idx - 1]
                    else:
                        var[idx] = var[idx + 1]

    for index, value in enumerate(var):
        if value != masked_value:
            result.append(value)
        else:
            masked_indices.append(index)

    # for value in var:
    #     if value != masked_value:
    #         result.append(value)

    return result, masked_indices

def get_min_max_from_nc_var(name, var):
    if name == 'DATE':
        the_min = datetime.datetime.strptime(str(var[0]),
                                             '%Y%m%d').date()
        the_max = datetime.datetime.strptime(str(var[-1]),
                                             '%Y%m%d').date()
    elif name == 'TIME':
        min_time_str = str(var[0])
        max_time_str = str(var[-1])
        the_min = sfmr_nc_time_converter(min_time_str)
        the_max = sfmr_nc_time_converter(max_time_str)
    elif name == 'LON':
        the_min = (min(var) + 360) % 360
        the_max = (max(var) + 360) % 360
    elif name == 'LAT':
        the_min = min(var)
        the_max = max(var)

    return the_min, the_max

def sfmr_nc_time_converter(time_str):
    # Only seconds
    if len(time_str) <= 2:
        the_time = datetime.datetime.strptime(
            time_str, '%S').time()
    elif len(time_str) == 3:
        the_time = datetime.datetime.strptime(
            f'0{time_str}', '%M%S').time()
    elif len(time_str) == 4:
        the_time = datetime.datetime.strptime(
            time_str, '%M%S').time()
    elif len(time_str) == 5:
        the_time = datetime.datetime.strptime(
            f'0{time_str}', '%H%M%S').time()
    elif len(time_str) == 6:
        the_time = datetime.datetime.strptime(
            time_str, '%H%M%S').time()
    else:
        logger.error('Length of SFMR TIME input is longer than 6')
        exit()

    return the_time

"""
def get_terminal_index_among_vars(side, vars, masked_value):
    terminal_indices = []
    for var in vars:
        one_var_termianl_indices = get_terminal_indices_of_nc_var(
            var, masked_value)
        terminal_indices.append(one_var_indices)

def get_terminal_indices_of_nc_var(var, masked_value):
    encounter_masked = False
    start_indices = []
    for idx, val in enumerate(var):
        if val == masked_value and not encounter_masked:
            encounter_masked = True
            start_indices.append(idx)
"""

def get_sfmr_track_and_windspd(nc_file_path, data_indices):
    """Get data points along SFMR track, averaged points' longtitude,
    averaged points' latitude and averaged points' wind speed.

    Parameters
    ----------
    nc_file_path : str
        Path of SFMR data file.
    data_indices : list of int
        The indices of data points in single SFMR data file that in
        the spatial and temporal window around TC center.

    Returns
    -------
    track : list of float tuple
        All points' coordinate along SFMR track that in the spatial and
        temporal window around TC center.
    new_avg_dts : list of datetime
        Datetime of averaged points along SFMR track that in the spatial
        and temporal window around TC center.
    new_avg_lons : list of float
        Longitude of averaged points along SFMR track that in the spatial
        and temporal window around TC center.
    new_avg_lats : list of float
        Latitude of averaged points along SFMR track that in the spatial
        and temporal window around TC center.
    new_avg_windspd : list of float
        Wind speed of averaged points along SFMR track that in the
        spatial and temporal window around TC center.

    """
    dataset = netCDF4.Dataset(nc_file_path)
    # VERY VERY IMPORTANT: netCDF4 auto mask all windspd which
    # faster than 1 m/s, so must disable auto mask
    dataset.set_auto_mask(False)
    vars = dataset.variables
    lons = vars['LON']
    lats = vars['LAT']
    wind = vars['SWS']
    track = []
    track_wind = []
    track_dt = []
    avg_dts = []
    avg_lons = []
    avg_lats = []
    avg_windspd = []

    # Get track
    for idx in data_indices:
        lon = (lons[idx] + 360) % 360
        lat = lats[idx]
        track.append((lon, lat))
        track_wind.append(wind[idx])
        track_dt.append(datetime.datetime.combine(
            sfmr_nc_converter('DATE', vars['DATE'][idx]),
            sfmr_nc_converter('TIME', vars['TIME'][idx])
        ))

    earliest_dt_of_track = min(track_dt)

    square_edge = 0.25
    half_square_edge = square_edge / 2
    # Split track with `square_edge` * `square_edge` degree squares
    # around SFMR points
    # For each point along track, there are two possibilities:
    # 1. center of square
    # 2. not center of square

    # First find all centers of square
    square_center_indices = []
    for idx, pt in enumerate(track):
        # Set start terminal along track as first square center
        if not idx:
            avg_lons.append(pt[0])
            avg_lats.append(pt[1])
            square_center_indices.append(idx)
        # Then search for next square center
        # If the difference between longitude or latitude is not smaller
        # than square edge, we encounter the center point of next square
        if (abs(pt[0] - avg_lons[-1]) >= square_edge
            or abs(pt[1] - avg_lats[-1]) >= square_edge):
            # Record the new square center
            avg_lons.append(pt[0])
            avg_lats.append(pt[1])
            square_center_indices.append(idx)

    # There may be some points left near the end terminal of track.
    # from which cannot select the square center using method above.
    # But they are out of the range of last square.
    # So we need to set the end terminal of track to be the last square
    # center.
    end_is_center = False
    for idx, pt in enumerate(track):
        if idx <= square_center_indices[-1]:
            continue
        if (abs(pt[0] - avg_lons[-1]) >= half_square_edge
            or abs(pt[1] - avg_lats[-1]) >= half_square_edge):
            # Make sure of the necessity of setting the end terminal as
            # the last square center
            end_is_center = True
            break

    if end_is_center:
        avg_lons.append(track[-1][0])
        avg_lats.append(track[-1][1])
        square_center_indices.append(len(track) - 1)

    # Save result of spliting into a list of list:
    # [[square_0_pt_0, square_0_pt_1, ...], [square_1_pt_0,
    # square_1_pt_1, ...], ...]
    square_groups = []
    for center in square_center_indices:
        square_groups.append([])

    for idx, pt in enumerate(track):
        try:
            min_dis = 999
            idx_of_nearest_center_idx = None
            for index_of_center_indices in range(
                len(square_center_indices)):
                #
                center_index = square_center_indices[
                    index_of_center_indices]
                tmp_center = track[center_index]
                lon_diff = abs(pt[0] - tmp_center[0])
                lat_diff = abs(pt[1] - tmp_center[1])
                if (lon_diff <= half_square_edge
                    and lat_diff <= half_square_edge):
                    dis = math.sqrt(math.pow(lon_diff, 2)
                                    + math.pow(lat_diff, 2))
                    if dis < min_dis:
                        min_dis = dis
                        idx_of_nearest_center_idx = \
                                index_of_center_indices
                    # This point along track is in one square

            if idx_of_nearest_center_idx is not None:
                square_groups[idx_of_nearest_center_idx].append(idx)
            else:
                continue
        except Exception as msg:
            breakpoint()
            exit(msg)

    if len(square_groups) != len(square_center_indices):
        logger.error(f"""Number of groups does not equal to """
                     f"""number of centers""")
        breakpoint()

    to_delete_empty_groups_indices = []
    # Average wind speed in square to the center point
    for index_of_center_indices in range(len(square_center_indices)):
        try:
            group_size = len(square_groups[index_of_center_indices])
            if not group_size:
                to_delete_empty_groups_indices.append(
                    index_of_center_indices)
                continue

            # Calculated the average datetime of each group
            seconds_shift_sum = 0
            for pt_index in square_groups[index_of_center_indices]:
                temporal_shift = (track_dt[pt_index]
                                  - earliest_dt_of_track)
                seconds_shift_sum += (temporal_shift.days * 24 * 3600
                                      + temporal_shift.seconds)
            avg_seconds_shift = seconds_shift_sum / group_size
            avg_dts.append(earliest_dt_of_track + datetime.timedelta(
                seconds=avg_seconds_shift))

            # Calculate the average wind speed of each group
            windspd_sum = 0
            for pt_index in square_groups[index_of_center_indices]:
                # Masked wind is smaller than 0
                if track_wind[pt_index] > 0:
                    windspd_sum += track_wind[pt_index]
            avg_windspd.append(windspd_sum / group_size)
        except Exception as msg:
            breakpoint()
            exit(msg)

    if len(to_delete_empty_groups_indices):
        new_avg_dts = []
        new_avg_lons = []
        new_avg_lats = []
        new_avg_windspd = []
        for i in range(len(avg_windspd)):
            if i not in to_delete_empty_groups_indices:
                new_avg_dts.append(avg_dts[i])
                new_avg_lons.append(avg_lons[i])
                new_avg_lats.append(avg_lats[i])
                new_avg_windspd.append(avg_windspd[i])

        del avg_dts
        del avg_lons
        del avg_lats
        del avg_windspd
        avg_dts = new_avg_dts
        avg_lons = new_avg_lons
        avg_lats = new_avg_lats
        avg_windspd = new_avg_windspd

    return track, avg_dts, avg_lons, avg_lats, avg_windspd

def draw_sfmr_windspd_and_track(the_class, fig, ax, tc_datetime,
                                sfmr_tracks, sfmr_lons, sfmr_lats,
                                sfmr_windspd, max_windspd):
    track_lons = []
    track_lats = []
    for single_track in sfmr_tracks:
        for pt in single_track:
            track_lons.append(pt[0])
            track_lats.append(pt[1])
        ax.plot(track_lons, track_lats, color='black',
                zorder=the_class.zorders['sfmr_track'])

    try:
        tracks_num = len(sfmr_windspd)
        for i in range(tracks_num):
            # for j in range(len(sfmr_windspd[i])):
            ax.scatter(sfmr_lons[i], sfmr_lats[i], s=60,
                       c=sfmr_windspd[i], cmap=plt.cm.rainbow,
                       vmin=0, vmax=max_windspd,
                       zorder=the_class.zorders['sfmr_point'],
                       edgecolors='black')
    except Exception as msg:
        breakpoint()
        exit(msg)

def interp_satel_era5_diff_mins_matrix(diff_mins):
    rows_num, cols_num = diff_mins.shape
    # Fill row by row
    for i in range(rows_num):
        row_valid_sum = 0
        row_valid_count = 0
        for j in range(cols_num):
            # In one-hour window
            if abs(diff_mins[i][j]) <= 30:
                row_valid_sum += diff_mins[i][j]
                row_valid_count += 1
        if not row_valid_count:
            continue
        row_valid_avg = int(row_valid_sum / row_valid_count)

        for j in range(cols_num):
            if abs(diff_mins[i][j]) > 30:
                diff_mins[i][j] = row_valid_avg

    # Fill column by column
    for j in range(cols_num):
        col_valid_sum = 0
        col_valid_count = 0
        for i in range(rows_num):
            # In one-hour window
            if abs(diff_mins[i][j]) <= 30:
                col_valid_sum += diff_mins[i][j]
                col_valid_count += 1
        if not col_valid_count:
            continue
        col_valid_avg = int(col_valid_sum / col_valid_count)

        for i in range(rows_num):
            if abs(diff_mins[i][j]) > 30:
                diff_mins[i][j] = col_valid_avg

    return diff_mins

def validate_with_sfmr(tgt_name, tc_dt, sfmr_dts, sfmr_lons,
                       sfmr_lats, sfmr_windspd, tgt_lons, tgt_lats,
                       tgt_windspd, tgt_mesh):
    """'tgt' is the abbreviation for word 'target'.

    """
    num_sfmr_tracks = len(sfmr_dts)
    num_tgt_lats, num_tgt_lons = tgt_windspd.shape

    grid_edge = 0.25
    half_grid_edge = grid_edge / 2
    max_min_dis = math.sqrt(half_grid_edge ** 2 + half_grid_edge ** 2)

    # 'tb' is the abbreviation for word 'table'
    tb_tgt_names = []
    tb_sfmr_dts = []
    tb_sfmr_lons = []
    tb_sfmr_lats = []
    tb_sfmr_windspd = []
    tb_tgt_dts = []
    tb_tgt_lons = []
    tb_tgt_lats = []
    tb_tgt_windspd = []
    tb_dis_minutes = []
    tb_dis_kms = []
    tb_windspd_bias = []

    # Traverse SFMR data points
    for t in range(num_sfmr_tracks):
        for i in range(len(sfmr_dts[t])):
            base_lon = sfmr_lons[t][i]
            base_lat = sfmr_lats[t][i]
            # Traverse tgt data points which have valid wind speed to
            # find the one closest to the SFMR data point
            min_dis = 999999.9
            min_dis_lat_idx = None
            min_dis_lon_idx = None
            min_dis_lat = None
            min_dis_lon = None
            try:
                for j in range(num_tgt_lats):
                    if not tgt_mesh:
                        tmp_lat = tgt_lats[j]
                    for k in range(num_tgt_lons):
                        # Skip if the data point from target source is
                        # invalid
                        if (tgt_windspd[j][k] is None
                            or tgt_windspd[j][k] <= 0):
                            continue
                        if not tgt_mesh:
                            tmp_lon = tgt_lons[k]
                        else:
                            tmp_lat = tgt_lats[j][k]
                            tmp_lon = tgt_lons[j][k]
                        # Calculate the distance between SFMR data point
                        # and data point from target source
                        dis = math.sqrt(
                            math.pow(base_lon - tmp_lon, 2)
                            + math.pow(base_lat - tmp_lat, 2))
                        if dis < min_dis:
                            min_dis = dis
                            min_dis_lat_idx = j
                            min_dis_lon_idx = k
            except Exception as msg:
                breakpoint()
                exit(msg)

            # Skip if there are no data point from target source in the
            # spatial window around SFMR data point
            if min_dis > max_min_dis:
                continue

            if tgt_mesh:
                min_dis_lat = tgt_lats[min_dis_lat_idx][min_dis_lon_idx]
                min_dis_lon = tgt_lons[min_dis_lat_idx][min_dis_lon_idx]
            else:
                min_dis_lat = tgt_lats[min_dis_lat_idx]
                min_dis_lon = tgt_lons[min_dis_lon_idx]
            min_dis_windspd = tgt_windspd[min_dis_lat_idx][
                min_dis_lon_idx]

            tb_tgt_names.append(tgt_name)
            tb_sfmr_dts.append(sfmr_dts[t][i])
            tb_sfmr_lons.append(base_lon)
            tb_sfmr_lats.append(base_lat)
            tb_sfmr_windspd.append(sfmr_windspd[t][i])
            tb_tgt_dts.append(tc_dt)
            tb_tgt_lons.append(min_dis_lon)
            tb_tgt_lats.append(min_dis_lat)
            tb_tgt_windspd.append(min_dis_windspd)

            # Temporal distance in minutes
            temporal_dis = tc_dt - sfmr_dts[t][i]
            tb_dis_minutes.append(temporal_dis.days * 24 * 60
                                  + temporal_dis.seconds / 60)
            # Spatial distance in kilo meters
            sfmr_pt = (base_lat,
                       longtitude_converter(base_lon, '360', '-180'))
            tgt_pt = (min_dis_lat,
                      longtitude_converter(min_dis_lon, '360', '-180'))
            tb_dis_kms.append(distance.distance(tgt_pt, sfmr_pt).km)
            # Bias of wind speed
            tb_windspd_bias.append(min_dis_windspd - sfmr_windspd[t][i])

    # Combine lists for table to pandas DataFrame
    res_dict = {
        'tgt_name': tb_tgt_names,
        'sfmr_dt': tb_sfmr_dts,
        'sfmr_lon': tb_sfmr_lons,
        'sfmr_lat': tb_sfmr_lats,
        'sfmr_windspd': tb_sfmr_windspd,
        'tgt_dt': tb_tgt_dts,
        'tgt_lon': tb_tgt_lons,
        'tgt_lat': tb_tgt_lats,
        'tgt_windspd': tb_tgt_windspd,
        'dis_minutes': tb_dis_minutes,
        'dis_kms': tb_dis_kms,
        'windspd_bias': tb_windspd_bias,
    }
    # Will datetime columns remain their type?
    res_df = pd.DataFrame(res_dict)

    return res_df

def get_bound_of_multiple_int(lims, interval):
    """lims[0] < lims[1]

    """
    bottom_lim = closest_multiple_int('bottom', lims[0], interval)
    top_lim = closest_multiple_int('top', lims[1], interval)

    return (bottom_lim, top_lim)

def closest_multiple_int(direction, value, interval):
    nearest = round(value / interval) * interval

    if direction == 'bottom':
        if nearest > value:
            nearest -= interval
    elif direction == 'top':
        if nearest < value:
            nearest += interval

    return int(nearest)
