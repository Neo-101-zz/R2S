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

unixOptions = 'p:a:er:c:si:'
# gnuOptions = ['extract', 'reg-dnn', 'reg-xgb', 'reg-dt',
#               'reg-hist', 'reg-normalization', 'compare',
#               'sfmr', 'ibtracs-wp', 'ibtracs-na']
gnuOptions = ['period=', 'area=', 'basin=', 'extract', 'reg=', 'compare=',
              'sfmr', 'ibtracs=']

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
    input_custom_area = False
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
    # evaluate given options
    for current_argument, current_value in arguments:
        if current_argument in ('-p', '--period'):
            input_custom_period = True
            period_parts = current_value.split(',')
            if len(period_parts) != 2:
                logger.error('Inputted period is wrong: need 2 parameters')
        elif current_argument in ('-a', '--area'):
            input_custom_area = True
            area_parts = current_value.split(',')
            if len(area_parts) != 4:
                logger.error('Inputted area is wrong: need 4 parameters')
        elif current_argument in ('-b', '--basin'):
            specify_basin = True
            basin_parts = current_value.split(',')
            if len(basin_parts) != 1:
                logger.error('Inputted basin is wrong: must 1 parameters')
            basin = basin_parts[0]
        elif current_argument in ('-e', '--extract'):
            do_extract = True
        elif current_argument in ('-r', '--reg'):
            do_regression = True
            reg_instructions = current_value.split(',')
        elif current_argument in ('-c', '--compare'):
            do_compare = True
            compare_instructions = current_value.split(',')
        elif current_argument in ('-s', '--sfmr'):
            do_sfmr = True
        elif current_argument in ('-i', '--ibtracs'):
            do_ibtracs = True

    if not specify_basin:
        logger.error('Must specify basin')
        exit()

    if input_custom_period:
        # Period parts
        # yyyy-mm-dd-HH-MM-SS
        period = [
            datetime.datetime.strptime(period_parts[0], '%Y-%m-%d-%H-%M-%S'),
            datetime.datetime.strptime(period_parts[1], '%Y-%m-%d-%H-%M-%S')
        ]
    else:
        period = [datetime.datetime(2015, 4, 1, 0, 0, 0),
                  datetime.datetime.now()]
    if input_custom_area:
        # Area parts
        custom_area = []
        for part in area_parts:
            custom_area.append(float(part))
    else:
        area = [-90, 90, 0, 360]

    # Period
    logger.info(f'Period: {period}')
    # Region
    logger.info(f'Region: {area}')
    # MySQL Server root password
    passwd = '399710'
    # Download and read
    try:
        if do_extract:
            extract_ = match_era5_smap.matchManager(
                CONFIG, period, area, basin, passwd, False)
        # sta = statistic.StatisticManager(CONFIG, period, area,
        #                                  passwd, save_disk=False)
        if do_compare:
            com_tc = compare_tc.TCComparer(CONFIG, period, area, basin,
                                           passwd, False,
                                           compare_instructions)
        # ccmp_ = ccmp.CCMPManager(CONFIG, period, area, passwd,
        #                          work_mode='fetch_and_compare')
        # era5_ = era5.ERA5Manager(CONFIG, period, area, passwd,
        #                          work=True, save_disk=False, 'scs',
        #                          'surface_all_vars')
        # isd_ = isd.ISDManager(CONFIG, period, area, passwd,
        #                       work_mode='fetch_and_read')
        # grid_ = grid.GridManager(CONFIG, area, passwd, run=True)
        # satel_scs_ = satel_scs.SCSSatelManager(CONFIG, period,
        #                                        area, passwd,
        #                                        save_disk=False,
        #                                        work=True)
        # coverage_ = coverage.CoverageManager(CONFIG, period,
        #                                      area, passwd)
        if do_ibtracs:
            ibtracs_ = ibtracs.IBTrACSManager(CONFIG, period, area, basin,
                                              passwd)
        # cwind_ = cwind.CwindManager(CONFIG, period, area, passwd)
        # stdmet_ = stdmet.StdmetManager(CONFIG, period, area, passwd)
        if do_sfmr:
            sfmr_ = sfmr.SfmrManager(CONFIG, period, area, passwd)
        # satel_ = satel.SatelManager(CONFIG, period, area, passwd,
        #                             save_disk=False)
        # compare_ = compare_offshore.CompareCCMPWithInStu(
        #     CONFIG, period, area, passwd)
        pass
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')

    test_period = [datetime.datetime(2013, 6, 6, 0, 0, 0),
                   datetime.datetime(2013, 6, 6, 23, 0, 0)]
    test_period = period
    try:
        if do_regression:
            regression_ = regression.Regression(
                CONFIG, period, test_period, area, basin, passwd, False,
                reg_instructions)
        # new_reg = reg_scs.NewReg(CONFIG, period, test_period,
        #                          area, passwd, save_disk=True)
        # ibtracs_ = ibtracs.IBTrACSManager(CONFIG, test_period,
        #                                   area, passwd)
        # hwind_ = hwind.HWindManager(CONFIG, test_period, area, passwd)
        # era5_ = era5.ERA5Manager(CONFIG, test_period, area, passwd,
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
