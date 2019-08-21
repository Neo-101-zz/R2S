# !/usr/bin/env python

"""Key is period and region.
BUT cannot verify data's region before download it.
So the key is period.

"""
from datetime import date

import load_config
import dl_ndbc
import dl_sfmr
import dl_satel
import rd_ndbc
import rd_sfmr
import rd_satel

def download_ndbc(CONFIG, period):
    """Download NDBC's buoys station's information and corresponding
    Continuous Wind Historical Data in given period.

    Parameters
    ----------
    CONFIG : dict
        Configuration. See config.yaml.
    period : list of datetime.date
        Two element list of datetime.date.  First element is start date
        and second element is end date.

    """
    years = [x for x in range(period[0].year, period[1].year+1)]
    stations = set()
    year_station = dl_ndbc.analysis_and_save_relation(CONFIG, years,
                                                      stations)
    dl_ndbc.download_station_info(CONFIG, stations)
    dl_ndbc.download_cwind_data(CONFIG, years, year_station)

def download_sfmr(CONFIG, period):
    years = [x for x in range(period[0].year, period[1].year+1)]
    year_hurr = dl_sfmr.gen_year_hurr(CONFIG, years)
    hit_times = dl_sfmr.download_sfmr_data(CONFIG, year_hurr, period)
    dl_sfmr.save_year_hurr(CONFIG, year_hurr, hit_times)

def download_satel(CONFIG, satel_name, period):
    if not dl_satel.check_satel_period(CONFIG, satel_name, period):
        return
    dl_satel.download_satel_data(CONFIG, satel_name, period)

def read_ndbc(CONFIG, region):
    rd_ndbc.gen_station_csv(CONFIG, region)
    rd_ndbc.read_cwind_data(CONFIG)

def read_sfmr(CONFIG):
    rd_sfmr.read_sfmr(CONFIG)

def read_ascat(CONFIG, region):
    pass

def read_qscat(CONFIG, region):
    pass

def read_wscat(CONFIG, region):
    pass

def test(CONFIG):
    period = dl_satel.input_period(CONFIG)
    download_ndbc(CONFIG, period)
    download_sfmr(CONFIG, period)
    download_satel(CONFIG, 'ascat', period)
    download_satel(CONFIG, 'qscat', period) 
    download_satel(CONFIG, 'wsat', period)

    # region = rd_ndbc.set_region(CONFIG)
    # read_ndbc(CONFIG, region)
    # read_sfmr(CONFIG)

if __name__ == '__main__':
    CONFIG = load_config.load_config()
    test(CONFIG)
