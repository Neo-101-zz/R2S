import math

def windspd(u_ms, v_ms):
    return 1.94384 * math.sqrt(u_ms**2 + v_ms**2)

# u_ms, v_ms = input().replace('  ', ' ').split(' ')
# print(windspd(float(u_ms), float(v_ms)))

from netCDF4 import Dataset
import numpy as np

MASKED = np.ma.core.masked

ds = Dataset('../data/satel/smap_ncs/RSS_smap_wind_daily_2018_06_30_v01.0.nc')
# ds.set_auto_mask(False)
vars = ds.variables
breakpoint()

max_wind = 0

for y in range(720):
    max = vars['wind'][y].max()
    if max > max_wind:
        max_wind = max

print(max_wind)

