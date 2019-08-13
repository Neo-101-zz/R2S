import numpy as np
import sys

from windsat_daily_v7 import WindSatDaily
import read_util

def read_windsat(filename, missing_val):
    dataset = WindSatDaily(filename, missing=missing_val)
    if not dataset.variables:
        sys.exit('file not found')
    return dataset

def main():
    lats = (30, 50)
    lons = (220, 242)
    missing_val = -999.0
    dataset = read_windsat('../data/satel/windsat/wsat_20030206v7.0.1.gz',
                           missing_val)
    read_util.show_dimensions(dataset)
    read_util.show_variables(dataset)
    read_util.show_validrange(dataset)
    windsats = read_util.cut_map('windsat', dataset, lats, lons, 2, 5, missing_val)

if __name__ == '__main__':
    main()
