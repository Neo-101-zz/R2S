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
        req = requests.head(url)
        if url.endswith('.gz'):
            try:
                if req.headers['Content-Type'].startswith('application'):
                    return True
                else:
                    return False
            except Exception as msg:
                breakpoint()
                exit(msg)
        # elif url.endswith('.nc'):
        else:
            return True
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
        logger.error(f'{table_fullname} not exists')
        return None

def add_column(engine, table_name, column):
    column_name = column.compile(dialect=engine.dialect)
    column_type = column.type.compile(engine.dialect)
    connection = engine.connect()
    result = connection.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type))
    connection.close()

def get_mysql_connector(the_class):
    DB_CONFIG = the_class.CONFIG['database']
    PROMPT = the_class.CONFIG['workflow']['prompt']
    DBAPI = DB_CONFIG['db_api']
    USER = DB_CONFIG['user']
    # password_ = input(PROMPT['input']['db_root_password'])
    password_ = the_class.db_root_passwd
    HOST = DB_CONFIG['host']
    PORT = DB_CONFIG['port']
    DB_NAME = DB_CONFIG['db_name']
    ARGS = DB_CONFIG['args']

    cnx = mysql.connector.connect(user=USER, password=password_,
                                  host=HOST, port=PORT, use_pure=True)
    return cnx

def setup_database(the_class, Base):
    DB_CONFIG = the_class.CONFIG['database']
    PROMPT = the_class.CONFIG['workflow']['prompt']
    DBAPI = DB_CONFIG['db_api']
    USER = DB_CONFIG['user']
    # password_ = input(PROMPT['input']['db_root_password'])
    password_ = the_class.db_root_passwd
    HOST = DB_CONFIG['host']
    PORT = DB_CONFIG['port']
    DB_NAME = DB_CONFIG['db_name']
    ARGS = DB_CONFIG['args']

    # the_class.cnx = mysql.connector.connect(user=USER, password=password_,
    #                                    host=HOST, use_pure=True)
    # the_class.cursor = the_class.cnx.cursor()
    # create_database(the_class.cnx, DB_NAME)
    # use_database(the_class.cnx, DB_NAME)
    mysql_connector = get_mysql_connector(the_class)
    create_database(mysql_connector, DB_NAME)
    use_database(mysql_connector, DB_NAME)
    mysql_connector.close()

    # Define the MySQL engine using MySQL Connector/Python
    connect_string = ('{0}://{1}:{2}@{3}:{4}/{5}?{6}'.format(
        DBAPI, USER, password_, HOST, PORT, DB_NAME, ARGS))
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

def get_tc_center(tc_row):
    lon_converted = tc_row.lon + 360 if tc_row.lon < 0 else tc_row.lon
    center = (lon_converted, tc_row.lat)

    return center

def draw_ibtracs_radii(ax, tc_row, zorders):
    center = get_tc_center(tc_row)
    tc_radii = get_radii_from_tc_row(tc_row)
    # radii_color = {34: 'yellow', 50: 'orange', 64: 'red'}
    radii_hatch = {34: '-', 50: 'x', 64: 'o'}
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
                    fill=False, hatch=radii_hatch[r]
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
    cols.append(Column('date_time', DateTime, nullable=False))
    cols.append(Column('x', Integer, nullable=False))
    cols.append(Column('y', Integer, nullable=False))
    cols.append(Column('lon', Float, nullable=False))
    cols.append(Column('lat', Float, nullable=False))
    cols.append(Column('datetime_x_y', String(20), nullable=False,
                       unique=True))

    return cols

def get_basic_satel_era5_columns():
    cols = []
    cols.append(Column('key', Integer, primary_key=True))
    cols.append(Column('satel_datetime', DateTime, nullable=False))
    cols.append(Column('era5_datetime', DateTime, nullable=False))
    cols.append(Column('x', Integer, nullable=False))
    cols.append(Column('y', Integer, nullable=False))
    cols.append(Column('lon', Float, nullable=False))
    cols.append(Column('lat', Float, nullable=False))
    cols.append(Column('satel_datetime_lon_lat', String(50), nullable=False,
                       unique=True))

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

def draw_SCS_basemap(the_class, ax):
    map = Basemap(llcrnrlon=the_class.lon1, llcrnrlat=the_class.lat1,
                  urcrnrlon=the_class.lon2, urcrnrlat=the_class.lat2,
                  ax=ax, resolution='h')

    map.drawcoastlines(zorder=the_class.zorders['coastlines'])
    map.drawmapboundary(fill_color='white',
                        zorder=the_class.zorders['mapboundary'])
    map.fillcontinents(color='grey', lake_color='white',
                       zorder=the_class.zorders['continents'])

    map.drawmeridians(np.arange(100, 126, 5), labels=[1, 0, 0, 1],
                      zorder=the_class.zorders['grid'])
    map.drawparallels(np.arange(0, 31, 5), labels=[1, 0 , 0, 1],
                      zorder=the_class.zorders['grid'])

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
    if mesh:
        X, Y = np.meshgrid(lons, lats)
    else:
        X, Y = lons, lats
    Z = windspd

    # windspd_levels = [5*x for x in range(1, 15)]
    windspd_levels = np.linspace(0, max_windspd, 10)

    # cs = ax.contour(X, Y, Z, levels=windspd_levels,
    #                 zorder=self.zorders['contour'], colors='k')
    # ax.clabel(cs, inline=1, colors='k', fontsize=10)
    try:
        cf = ax.contourf(X, Y, Z, levels=windspd_levels,
                         zorder=wind_zorder,
                         cmap=plt.cm.rainbow)
    except Exception as msg:
        breakpoint()
        exit(msg)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(cf, cax=cax, orientation='vertical', format='%.1f')

def draw_ccmp_windspd(the_class, fig, ax, dt, lons, lats, windspd,
                      max_windspd):
    draw_SCS_basemap(the_class, ax)

    draw_windspd_with_contourf(fig, ax, lons, lats, windspd,
                               the_class.zorders['contourf'],
                               max_windspd, mesh=True)

def draw_windspd(the_class, fig, ax, dt, lons, lats, windspd,
                 max_windspd, mesh):
    draw_SCS_basemap(the_class, ax)

    draw_windspd_with_contourf(fig, ax, lons, lats, windspd,
                               the_class.zorders['contourf'],
                               max_windspd, mesh)

def draw_era5_wind(the_class, fig, ax, dt, lons, lats, winspd,
                   max_windspd):
    draw_SCS_basemap(the_class, ax)

    draw_windspd_with_contourf(fig, ax, lons, lats, windspd,
                               the_class.zorders['contourf'],
                               max_windspd, mesh=False)

def get_nearest_element_and_index(list_, element):
    ae = [abs(element - x) for x in list_]
    match_index = ae.index(min(ae))

    return list_[match_index], match_index

def get_xyz_of_ccmp_windspd(ccmp_file_path, dt, region):
    ccmp_hours = [0, 6, 12, 18]
    if dt.hour not in ccmp_hours:
        return None, None, None
    else:
        hour_idx = ccmp_hours.index(dt.hour)

    spa_resolu = 0.25

    ccmp_lats = [y * spa_resolu - 78.375 for y in range(628)]
    ccmp_lons = [x * spa_resolu + 0.125 for x in range(1440)]

    region[0], lat1_idx = get_nearest_element_and_index(ccmp_lats,
                                                        region[0])
    region[1], lat2_idx = get_nearest_element_and_index(ccmp_lats,
                                                        region[1])
    region[2], lon1_idx = get_nearest_element_and_index(ccmp_lons,
                                                        region[2])
    region[3], lon2_idx = get_nearest_element_and_index(ccmp_lons,
                                                        region[3])
    lons = list(np.arange(region[2], region[3] + 0.5 * spa_resolu,
                          spa_resolu))
    lats = list(np.arange(region[0], region[1] + 0.5 * spa_resolu,
                          spa_resolu))
    lons = [round(x, 2) for x in lons]
    lats = [round(y, 2) for y in lats]
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
                    windspd[y][x] = 0
            except Exception as msg:
                breakpoint()
                exit(msg)

    return lons, lats, windspd

def get_xyz_of_era5_windspd(era5_file_path, dt, region):
    grbidx = pygrib.index(era5_file_path, 'dataTime')
    hourtime = dt.hour * 100
    selected_grbs = grbidx.select(dataTime=hourtime)

    u_wind = None
    v_wind = None

    for grb in selected_grbs:
        name = grb.name.replace(" ", "_").lower()

        data, lats, lons = grb.data(region[0], region[1],
                                    region[2], region[3])
        data = np.flip(data, 0)
        lats = np.flip(lats, 0)
        lons = np.flip(lons, 0)

        if name == 'u_component_of_wind':
            u_wind = data
        elif name == 'v_component_of_wind':
            v_wind = data

    lats_num, lons_num = u_wind.shape
    windspd = np.ndarray(shape=u_wind.shape, dtype=float)
    for y in range(lats_num):
        for x in range(lons_num):
            try:
                if not bool(globe.is_land(lats[y][x], lons[y][x])):
                    windspd[y][x] = math.sqrt(u_wind[y][x] ** 2
                                              + v_wind[y][x] ** 2)
                else:
                    windspd[y][x] = 0
            except Exception as msg:
                breakpoint()
                exit(msg)

    return lons, lats, windspd

