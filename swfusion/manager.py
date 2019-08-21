from datetime import date

import cwind
import load_config
import utils

def work_flow():
    # CONFIG
    CONFIG = load_config.load_config()
    # Period
    # period = utils.input_period(CONFIG)
    period = [date(2007, 3, 1), date(2007, 3, 3)]
    # Region
    # region = utils.input_region(CONFIG)
    region = [-90, 90, 0, 360]
    # Download
    cwind_ = cwind.CwindManager(CONFIG, period, region)
    # Read
    # Match
    # Validate
    # Fusion

if __name__ == '__main__':
    work_flow()
