from datetime import datetime
import logging
import os

import cwind
import sfmr
import satel
import load_configs
import utils

def work_flow():
    # CONFIG
    CONFIG = load_configs.load_config()
    os.makedirs(CONFIG['logging']['dir'], exist_ok=True)
    load_configs.setup_logging()
    breakpoint()
    # Period
    # period = utils.input_period(CONFIG)
    period = [datetime(2008, 7, 11, 0, 0, 0),
              datetime(2008, 7, 14, 23, 59, 59)]
    # Region
    # region = utils.input_region(CONFIG)
    region = [-90, 90, 0, 360]
    # Download and read
    # cwind_ = cwind.CwindManager(CONFIG, period, region)
    # sfmr_ = sfmr.SfmrManager(CONFIG, period, region)
    satel_ = satel.SatelManager(CONFIG, period, region)
    # Match
    # Validate
    # Fusion

if __name__ == '__main__':
    work_flow()
