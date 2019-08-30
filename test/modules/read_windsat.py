import numpy as np
import sys

import read_util

def test():
    lats = (30, 50)
    lons = (220, 242)
    missing_val = -999.0
    dataset = read_util.read_daily_satel(
        'wsat', 
        '../data/satel/windsat/wsat_20030206v7.0.1.gz', missing_val)
    read_util.show_dimensions(dataset)
    read_util.show_variables(dataset)
    read_util.show_validrange(dataset)
    windsats = read_util.cut_map('wsat', dataset, lats, lons, 2, 5, missing_val)

if __name__ == '__main__':
    test()
