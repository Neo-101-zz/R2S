from datetime import date
import datetime
import os
import signal
import sys
import pickle

import numpy as np
from urllib import request
import requests
import progressbar
import mysql.connector
from mysql.connector import errorcode
import netCDF4
import bytemaps
from sqlalchemy import Table, Column, Integer, Float, String, DateTime, MetaData
from sqlalchemy.orm import mapper
from sqlalchemy import tuple_

from ascat_daily import ASCATDaily
from quikscat_daily_v4 import QuikScatDaily
from windsat_daily_v7 import WindSatDaily

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

def check_period(temporal, period):
    if type(temporal) is datetime.date:
        period = [x.date() for x in period]
    elif type(temporal) is datetime.time:
        period = [x.time() for x in period]
    elif type(temporal) is not datetime.datetime:
        exit('[Error] Type of inputted temporal variable should be ' \
             'datetime.date or datetime.time or datetime.datetime')
    start = period[0]
    end = period[1]
    if temporal < start or temporal > end or start > end:
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

    for i in range(length):
        if i % 5000 == 0:
            print('Row %d' % i)
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

    return whole_table

def is_missing(var, missing):
    if var is missing:
        return True
    return False

def convert_dtype(masked_array):
    if masked_array is np.ma.core.masked:
        return None
    for dtype in [np.int32, np.int64]:
        if np.issubdtype(masked_array.dtype, dtype):
            return int(masked_array)
    for dtype in [np.float32, np.float64]:
        if np.issubdtype(masked_array.dtype, dtype):
            return float(masked_array)

def create_table_from_netcdf(engine, nc_file, table_name, session,
                             skip_vars=None, notnull_vars=None,
                             unique_vars=None, custom_cols=None):
    # Sort custom_cols by column indices
    tmp = custom_cols
    custom_cols = dict()
    for i in sorted(tmp.keys()):
        custom_cols[i] = tmp[i]

    dataset = netCDF4.Dataset(nc_file)
    vars = dataset.variables

    class Netcdf(object):
        pass

    cols = []
    key = Column('key', Integer(), primary_key=True)
    cols.append(key)
    index = 1
    for var_name in vars.keys():
        while index in custom_cols:
            cols.append(custom_cols[index])
            index += 1
        if var_name in skip_vars:
            continue
        nullable = True if var_name not in notnull_vars else False
        unique = False if var_name not in unique_vars else True
        cols.append(nc2sacol(vars, var_name, nullable, unique))
        index += 1
    # In case that there is one custom col to insert into tial of row
    if index in custom_cols:
        cols.append(custom_cols[index])
        index += 1
    metadata = MetaData(bind=engine)
    t = Table(table_name, metadata, *cols)
    metadata.create_all()
    mapper(Netcdf, t)
    session.commit()

    return Netcdf

def nc2sacol(vars, var_name, nullable=True, unique=False):
    var_dtype = vars[var_name].dtype
    column = None
    for dtype in [np.int32, np.int64]:
        if np.issubdtype(var_dtype, dtype):
            return Column(var_name, Integer(), nullable=nullable,
                          unique=unique)
    for dtype in [np.float32, np.float64]:
        if np.issubdtype(var_dtype, dtype):
            return Column(var_name, Float(), nullable=nullable,
                          unique=unique)

def bulk_insert_avoid_duplicate_unique(total_sample, batch_size,
                                       table_class, unique_cols,
                                       session):
    """
    Bulkly insert into a table which has unique columns.

    """
    while total_sample:
        batch = total_sample[0:batch_size]
        total_sample = total_sample[batch_size:]

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
        if inserts:
            session.add_all(inserts)

    session.commit()

def create_table_from_bytemap(engine, satel_name, bytemap_file, table_name, 
                              session, skip_vars=None, notnull_vars=None,
                              unique_vars=None, custom_cols=None):
    # Sort custom_cols by column indices
    tmp = custom_cols
    custom_cols = dict()
    for i in sorted(tmp.keys()):
        custom_cols[i] = tmp[i]

    dataset = read_daily_satel(satel_name, bytemap_file)
    vars = dataset.variables

    class Satel(object):
        pass

    cols = []
    key = Column('key', Integer(), primary_key=True)
    cols.append(key)
    index = 1
    for var_name in vars.keys():
        while index in custom_cols:
            cols.append(custom_cols[index])
            index += 1
        if var_name in skip_vars:
            continue
        nullable = True if var_name not in notnull_vars else False
        unique = False if var_name not in unique_vars else True
        cols.append(bm2sacol(vars, var_name, nullable, unique))
        index += 1
    # In case that there is one custom col to insert into tial of row
    if -1 in custom_cols:
        cols.append(custom_cols[-1])

    metadata = MetaData(bind=engine)
    t = Table(table_name, metadata, *cols)
    metadata.create_all()
    mapper(Satel, t)
    session.commit()

def bm2sacol(vars, var_name, nullable=True, unique=False):
    var_dtype = vars[var_name].dtype
    column = None
    for dtype in [np.int32, np.int64]:
        if np.issubdtype(var_dtype, dtype):
            return Column(var_name, Integer(), nullable=nullable,
                          unique=unique)
    for dtype in [np.float32, np.float64]:
        if np.issubdtype(var_dtype, dtype):
            return Column(var_name, Float(), nullable=nullable,
                          unique=unique)

def read_daily_satel(satel_name, file_path, missing_val=-999.0):
    if satel_name == 'ascat':
        dataset = ASCATDaily(file_path, missing=missing_val)
    elif satel_name == 'qscat':
        dataset = QuikScatDaily(file_path, missing=missing_val)
    elif satel_name == 'wsat':
        dataset = WindSatDaily(file_path, missing=missing_val)
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
