import sys

import read_util

def test():
    lats = (30, 50)
    lons = (220, 242)
    missing_val = -999.0
    dataset1 = read_util.read_daily_satel(
        'ascat',
        '../data/satel/ascat/ascat_20070301_v02.1.gz', missing_val)
    read_util.show_dimensions(dataset1)
    read_util.show_variables(dataset1)
    read_util.show_validrange(dataset1)
    print()
    # (2, 720, 1440)
    print(dataset1.variables['scatflag'].shape)
    ascats = read_util.cut_map('ascat', dataset1, lats, lons, 2007,
                               3, 1, missing_val)
    # 3620
    print(len(ascats))
    # 0.0
    print(ascats[10]['rain'])

if __name__ == '__main__':
    test()
