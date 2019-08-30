import sys
from datetime import date

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

import read_util

def test():
    lats = (30, 50)
    lons = (220, 242)
    missing_val = -999.0
    date_ = date(1999, 7, 19)

    date_str = date_.strftime('%Y/%m/%d')
    file_name = date_.strftime('qscat_%Y%m%dv4.gz')

    dataset = read_util.read_daily_satel(
        'qscat', '../data/satel/qscat/' + file_name, missing_val)
    read_util.show_dimensions(dataset)
    read_util.show_variables(dataset)
    read_util.show_validrange(dataset)
    test = read_util.cut_map('qscat', dataset, lats, lons, 1999,
                               7, 19, missing_val)
    qscat = read_util.narrow_map(dataset, lats, lons)
    # qscat['wspd'][0].shape: (80, 88)
    print(qscat['wspd'][0].shape)
    exit(0)

    fig = plt.figure(1)
    m = Basemap(projection='mill', llcrnrlon=lons[0], urcrnrlon=lons[1], 
                llcrnrlat=lats[0], urcrnrlat=lats[1], resolution='l')
    aximg = m.imshow(qscat['wspd'][0], vmin=0, vmax=25)
    m.drawcoastlines(linewidth=0.25)
    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='#ddaa66', lake_color='#ddaa66')
    parallels = np.arange(lats[0], lats[1], 6.)
    meridians = np.arange(lons[0], lons[1], 7.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6.5, linewidth=0.8)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6.5, linewidth=0.8)
    cbar = fig.colorbar(aximg, orientation='vertical')
    cbar.set_label('m/s')
    plt.title('10-m Surface Wind Speed (m/s) of QuickScat: ' + date_str)
    plt.savefig(date_.strftime('qscat_%Y%m%d.png'))

if __name__ == '__main__':
    test()
