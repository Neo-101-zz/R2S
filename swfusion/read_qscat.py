import sys
from datetime import date

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

from quikscat_daily_v4 import QuikScatDaily
import read_util

def read_qscat(filename, missing_val):
    dataset = QuikScatDaily(filename, missing=missing_val)
    if not dataset.variables:
        sys.exit('file not found')
    return dataset

def narrow_map(dataset, lats, lons):
    # llcrnr: lower left hand corner
    # urcrnr: upper right hand corner
    llcrnrlat, urcrnrlat = read_util.find_index(lats, 'lat')
    llcrnrlon, urcrnrlon = read_util.find_index(lons, 'lon')

    map = {}
    map['wspd'] = []
    map['wdir'] = []
    map['rain'] = []
    map['time'] = []
    # iasc = 0 (morning, descending passes)
    # iasc = 1 (evening, ascending passes)
    iasc = [0, 1]
    vars = dataset.variables
    for i in iasc:
        wspd = vars['windspd'][i][llcrnrlat:urcrnrlat+1, llcrnrlon:urcrnrlon+1]
        wdir = vars['winddir'][i][llcrnrlat:urcrnrlat+1, llcrnrlon:urcrnrlon+1]
        rain = vars['scatflag'][i][llcrnrlat:urcrnrlat+1, llcrnrlon:urcrnrlon+1]
        time = vars['mingmt'][i][llcrnrlat:urcrnrlat+1, llcrnrlon:urcrnrlon+1]
        map['wspd'].append(wspd)
        map['wdir'].append(wdir)
        map['rain'].append(rain)
        map['time'].append(time)

    return map

def main():
    lats = (30, 50)
    lons = (220, 242)
    missing_val = -999.0
    date_ = date(1999, 7, 19)

    date_str = date_.strftime('%Y/%m/%d')
    file_name = date_.strftime('qscat_%Y%m%dv4.gz')

    dataset = read_qscat('../data/satel/qscat/' + file_name, missing_val)
    read_util.show_dimensions(dataset)
    read_util.show_variables(dataset)
    read_util.show_validrange(dataset)
    qscat = narrow_map(dataset, lats, lons)
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
    main()
