import sys

from ascat_daily import ASCATDaily
import read_util

def read_ascat(filename, missing_val):
    dataset = ASCATDaily(filename, missing=missing_val)
    if not dataset.variables:
        sys.exit('file not found')
    return dataset


def main():
    lats = (30, 50)
    lons = (220, 242)
    missing_val = -999.0
    dataset1 = read_ascat('../data/satel/ascat/ascat_20070301_v02.1.gz',
                         missing_val)
    read_util.show_dimensions(dataset1)
    read_util.show_variables(dataset1)
    read_util.show_validrange(dataset1)
    print()
    # (2, 720, 1440)
    print(dataset1.variables['scatflag'].shape)
    ascats = read_util.cut_map('ascat', dataset1, lats, lons,
                               3, 1, missing_val)
    # 3620
    print(len(ascats))
    # 0.0
    print(ascats[10]['rain'])

if __name__ == '__main__':
    main()
