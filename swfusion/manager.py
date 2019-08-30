from datetime import datetime
import logging
import os

import cwind
import sfmr
import satel
import load_configs
import utils

def work_flow():
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
    period = [datetime(2008, 7, 12, 0, 0, 0),
              datetime(2008, 7, 12, 1, 0, 0)]
    logger.info(f'Period: {period}')
    # Region
    # region = utils.input_region(CONFIG)
    region = [-90, 90, 0, 360]
    logger.info(f'Region: {region}')
    # MySQL Server root password
    passwd = '39cnj971hw-'
    # Download and read
    try:
        cwind_ = cwind.CwindManager(CONFIG, period, region, passwd)
        sfmr_ = sfmr.SfmrManager(CONFIG, period, region, passwd)
        satel_ = satel.SatelManager(CONFIG, period, region, passwd)
    except Exception as msg:
        logger.exception('Exception occured when downloading and reading')
    # Match
    # Validate
    # Fusion

if __name__ == '__main__':
    work_flow()
