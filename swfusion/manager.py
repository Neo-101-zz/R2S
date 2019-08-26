from datetime import datetime

import cwind
import sfmr
import satel
import load_config
import utils

def work_flow():
    # CONFIG
    CONFIG = load_config.load_config()
    # Period
    # period = utils.input_period(CONFIG)
    period = [datetime(2007, 8, 31, 0, 0, 0),
              datetime(2007, 9, 3, 23, 59, 59)]
    # Region
    # region = utils.input_region(CONFIG)
    region = [-90, 90, 0, 360]
    # Download
    # cwind_ = cwind.CwindManager(CONFIG, period, region)
    # sfmr_ = sfmr.SfmrManager(CONFIG, period, region)
    satel_ = satel.SatelManager(CONFIG, period, region)
    # Read
    # Match
    # Validate
    # Fusion

if __name__ == '__main__':
    work_flow()
