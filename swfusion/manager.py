"""Manager of tropical cyclone ocean surface wind reanalysis system.

"""
from datetime import datetime
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

unixOptions = "er"
gnuOptions = ["extract", "regression"]

def work_flow():
    """The work flow of blending several TC OSW.
    """
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

    do_extract = False
    do_regression = False
    # evaluate given options
    for current_argument, current_value in arguments:
        if current_argument in ("-e", "--extract"):
            do_extract = True
        elif current_argument in ("-r", "--regression"):
            do_regression = True

    load_configs.setup_logging()
    logger = logging.getLogger(__name__)
    # CONFIG
    try:
        CONFIG = load_configs.load_config()
    except Exception as msg:
        logger.exception('Exception occurred when loading config.')
    os.makedirs(CONFIG['logging']['dir'], exist_ok=True)
    # Period
    train_period = [datetime(2016, 10, 17, 0, 0, 0),
                    datetime(2019, 10, 1, 0, 0, 0)]
    logger.info(f'Period: {train_period}')
    # Region
    region = [0, 30, 98, 125]
    logger.info(f'Region: {region}')
    # MySQL Server root password
    passwd = '399710'
    # Download and read
    try:
        if do_extract:
            extract_ = match_era5_smap.matchManager(
                CONFIG, train_period, region, passwd, False)
        # sta = statistic.StatisticManager(CONFIG, train_period, region,
        #                                  passwd, save_disk=False)
        # com_tc = compare_tc.TCComparer(CONFIG, train_period, region,
        #                                passwd, save_disk=False)
        # ccmp_ = ccmp.CCMPManager(CONFIG, train_period, region, passwd,
        #                          work_mode='fetch_and_compare')
        # era5_ = era5.ERA5Manager(CONFIG, train_period, region, passwd,
        #                          work=True, save_disk=False, 'scs',
        #                          'surface_all_vars')
        # isd_ = isd.ISDManager(CONFIG, train_period, region, passwd,
        #                       work_mode='fetch_and_read')
        # grid_ = grid.GridManager(CONFIG, region, passwd, run=True)
        # satel_scs_ = satel_scs.SCSSatelManager(CONFIG, train_period,
        #                                        region, passwd,
        #                                        save_disk=False,
        #                                        work=True)
        # coverage_ = coverage.CoverageManager(CONFIG, train_period,
        #                                      region, passwd)
        # ibtracs_ = ibtracs.IBTrACSManager(CONFIG, train_period, region, passwd)
        # cwind_ = cwind.CwindManager(CONFIG, train_period, region, passwd)
        # stdmet_ = stdmet.StdmetManager(CONFIG, train_period, region, passwd)
        # sfmr_ = sfmr.SfmrManager(CONFIG, train_period, region, passwd)
        # satel_ = satel.SatelManager(CONFIG, train_period, region, passwd,
        #                             save_disk=False)
        # compare_ = compare_offshore.CompareCCMPWithInStu(
        #     CONFIG, train_period, region, passwd)
        pass
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')

    test_period = [datetime(2013, 6, 6, 0, 0, 0),
                   datetime(2013, 6, 6, 23, 0, 0)]
    test_period = train_period
    try:
        if do_regression:
            regression_ = regression.Regression(CONFIG, train_period,
                                                test_period, region,
                                                passwd, save_disk=False)
        # new_reg = reg_scs.NewReg(CONFIG, train_period, test_period,
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
