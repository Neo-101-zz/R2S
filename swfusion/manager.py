"""Manager of tropical cyclone ocean surface wind reanalysis system.

"""
from datetime import datetime
import logging
import os

import cwind
import stdmet
import sfmr
import satel
import compare_offshore
import ibtracs
import era5
import hwind
import load_configs
import utils

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
    # Period
    # period = utils.input_period(CONFIG)
    # period = [datetime(1996, 10, 6, 0, 0, 0),
    #           datetime(1996, 10, 6, 23, 59, 59)]

    # period = [datetime(2016, 5, 27, 0, 0, 0),
    #           datetime(2016, 5, 27, 23, 59, 59)]
    period = [datetime(2018, 6, 30, 0, 0, 0),
              datetime(2018, 7, 2, 18, 0, 0)]
    logger.info(f'Period: {period}')
    # Region
    # region = utils.input_region(CONFIG)
    region = [-90, 90, 0, 360]
    logger.info(f'Region: {region}')
    # Spatial and temporal window size
    spatial_window = 0.125 # degree
    temporal_window = 5*60 # second
    # MySQL Server root password
    passwd = '39cnj971hw-'
    # Download and read
    try:
        # ibtracs_ = ibtracs.IBTrACSManager(CONFIG, period, region, passwd)
        # hwind_ = hwind.HWindManager(CONFIG, period, region, passwd)
        # era5_ = era5.ERA5Manager(CONFIG, period, region, passwd)
        # cwind_ = cwind.CwindManager(CONFIG, period, region, passwd)
        # stdmet_ = stdmet.StdmetManager(CONFIG, period, region, passwd)
        # sfmr_ = sfmr.SfmrManager(CONFIG, period, region, passwd)
        satel_ = satel.SatelManager(CONFIG, period, region, passwd,
                                    spatial_window, temporal_window)
        # compare_ = compare_offshore.CompareCCMPWithInStu(
        #     CONFIG, period, region, passwd)
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')
    # Match
    # Validate
    # Fusion

if __name__ == '__main__':
    work_flow()
