"""Manager of tropical cyclone ocean surface wind reanalysis system.

"""
import datetime
import getopt
import logging
import os
import sys

import regression
import cwind
import stdmet
import sfmr
import compare_offshore
import ibtracs
import era5
import hwind
import load_configs
import utils
import satel_scs
import grid
import coverage
import isd
import reg_scs
import ccmp
import compare_tc
import statistic
import match_era5_smap
import validate

unixOptions = 'p:r:eg:c:siv'
# gnuOptions = ['extract', 'reg-dnn', 'reg-xgb', 'reg-dt',
#               'reg-hist', 'reg-normalization', 'compare',
#               'sfmr', 'ibtracs-wp', 'ibtracs-na']
gnuOptions = ['period=', 'region=', 'basin=', 'extract', 'reg=',
              'compare=', 'sfmr', 'ibtracs', 'validate']

def work_flow():
    """The work flow of blending several TC OSW.
    """
    load_configs.setup_logging()
    logger = logging.getLogger(__name__)
    # CONFIG
    try:
        CONFIG = load_configs.load_config()
    except Exception as msg:
        logger.exception('Exception occurred when loading config.')
    os.makedirs(CONFIG['logging']['dir'], exist_ok=True)

    # read commandline arguments, first
    full_cmd_arguments = sys.argv

    # - further arguments
    argument_list = full_cmd_arguments[1:]

    try:
        arguments, values = getopt.getopt(argument_list, unixOptions,
                                          gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))
        sys.exit(2)

    input_custom_period = False
    input_custom_region = False
    specify_basin = False
    basin = None
    do_extract = False
    do_regression = False
    reg_instructions = None
    do_compare = False
    do_sfmr = False
    sfmr_instructions = None
    do_ibtracs = False
    ibtracs_instructions = None
    do_validation = False
    # evaluate given options
    for current_argument, current_value in arguments:
        if current_argument in ('-p', '--period'):
            input_custom_period = True
            period_parts = current_value.split(',')
            if len(period_parts) != 2:
                logger.error((f"""Inputted period is wrong: """
                              f"""need 2 parameters"""))
        elif current_argument in ('-r', '--region'):
            input_custom_region = True
            region_parts = current_value.split(',')
            if len(region_parts) != 4:
                logger.error((f"""Inputted region is wrong: """
                              f"""need 4 parameters"""))
        elif current_argument in ('-b', '--basin'):
            specify_basin = True
            basin_parts = current_value.split(',')
            if len(basin_parts) != 1:
                logger.error('Inputted basin is wrong: must 1 parameters')
            basin = basin_parts[0]
        elif current_argument in ('-e', '--extract'):
            do_extract = True
        elif current_argument in ('-g', '--reg'):
            do_regression = True
            reg_instructions = current_value.split(',')
        elif current_argument in ('-c', '--compare'):
            do_compare = True
            compare_instructions = current_value.split(',')
        elif current_argument in ('-s', '--sfmr'):
            do_sfmr = True
        elif current_argument in ('-i', '--ibtracs'):
            do_ibtracs = True
        elif current_argument in ('-v', '--validate'):
            do_validation = True

    if not specify_basin:
        logger.error('Must specify basin')
        exit()

    if input_custom_period:
        # Period parts
        # yyyy-mm-dd-HH-MM-SS
        period = [
            datetime.datetime.strptime(period_parts[0],
                                       '%Y-%m-%d-%H-%M-%S'),
            datetime.datetime.strptime(period_parts[1],
                                       '%Y-%m-%d-%H-%M-%S')
        ]
    else:
        period = [datetime.datetime(2015, 4, 1, 0, 0, 0),
                  datetime.datetime.now()]
    train_test_split_dt = datetime.datetime(2019, 1, 1, 0, 0, 0)

    if input_custom_region:
        # Area parts
        custom_region = []
        for part in region_parts:
            custom_region.append(float(part))
    else:
        region = [-90, 90, 0, 360]

    # Period
    logger.info(f'Period: {period}')
    # Region
    logger.info(f'Region: {region}')
    # MySQL Server root password
    passwd = '399710'
    # Download and read
    try:
        if do_validation:
            validate.ValidationManager(CONFIG, period, basin)
        if do_extract:
            extract_ = match_era5_smap.matchManager(
                CONFIG, period, region, basin, passwd, False)
        if do_regression:
            regression_ = regression.Regression(
                CONFIG, period, train_test_split_dt, region, basin,
                passwd, False, reg_instructions)
        # sta = statistic.StatisticManager(CONFIG, period, region,
        #                                  passwd, save_disk=False)
        if do_compare:
            com_tc = compare_tc.TCComparer(CONFIG, period, region, basin,
                                           passwd, False,
                                           compare_instructions)
        # ccmp_ = ccmp.CCMPManager(CONFIG, period, region, passwd,
        #                          work_mode='fetch_and_compare')
        # era5_ = era5.ERA5Manager(CONFIG, period, region, passwd,
        #                          work=True, save_disk=False, 'scs',
        #                          'surface_all_vars')
        # isd_ = isd.ISDManager(CONFIG, period, region, passwd,
        #                       work_mode='fetch_and_read')
        # grid_ = grid.GridManager(CONFIG, region, passwd, run=True)
        # satel_scs_ = satel_scs.SCSSatelManager(CONFIG, period,
        #                                        region, passwd,
        #                                        save_disk=False,
        #                                        work=True)
        # coverage_ = coverage.CoverageManager(CONFIG, period,
        #                                      region, passwd)
        if do_ibtracs:
            ibtracs_ = ibtracs.IBTrACSManager(CONFIG, period, region,
                                              basin, passwd)
        # cwind_ = cwind.CwindManager(CONFIG, period, region, passwd)
        # stdmet_ = stdmet.StdmetManager(CONFIG, period, region, passwd)
        if do_sfmr:
            sfmr_ = sfmr.SfmrManager(CONFIG, period, region, passwd)
        # satel_ = satel.SatelManager(CONFIG, period, region, passwd,
        #                             save_disk=False)
        # compare_ = compare_offshore.CompareCCMPWithInStu(
        #     CONFIG, period, region, passwd)
        pass
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')

    try:
        # new_reg = reg_scs.NewReg(CONFIG, period, test_period,
        #                          region, passwd, save_disk=True)
        # ibtracs_ = ibtracs.IBTrACSManager(CONFIG, test_period,
        #                                   region, passwd)
        # hwind_ = hwind.HWindManager(CONFIG, test_period, region, passwd)
        # era5_ = era5.ERA5Manager(CONFIG, test_period, region, passwd,
        #                          work=True, save_disk=False)
        pass
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')

    logger.info('SWFusion complete.')
    # Match
    # Validate
    # Fusion

if __name__ == '__main__':
    work_flow()
