import math

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

"""
x = [100.0, 100.25]
y = [20.0, 20.25]
z = [[16, 8], [4, 12]]
f = interpolate.interp2d(x, y, z)

# xnew = np.arange(100.0, 100.26, 0.05)
xnew = [x * 0.05 + 100.0 for x in range(6)]
# ynew = np.arange(20.0, 20.26, 0.05)
ynew = [y * 0.05 + 20.0 for y in range(6)]
znew = f(xnew, ynew)
breakpoint()
half_cell_edge = 0.025

plt.imshow(znew, aspect='equal', origin='lower',
           extent=[100.00 - half_cell_edge, 100.25 + half_cell_edge,
                   20.00 - half_cell_edge, 20.25 + half_cell_edge])
plt.colorbar(orientation='vertical')
plt.xticks(xnew)
plt.yticks(ynew)
plt.show()
"""

x = [114.0, 114.25]
y = [22.0, 22.25]
z = [[-5.1975250244140625e-05, -2.9981136322021484e-05], [-5.46574592590332e-05, -3.8564205169677734e-05]]
f = interpolate.interp2d(x, y, z)

# xnew = np.arange(100.0, 100.26, 0.05)
xnew = [x * 0.05 + 114.0 for x in range(6)]
# ynew = np.arange(20.0, 20.26, 0.05)
ynew = [y * 0.05 + 22.0 for y in range(6)]
znew = f(xnew, ynew)
breakpoint()
half_cell_edge = 0.025

plt.imshow(znew, aspect='equal', origin='lower',
           extent=[100.00 - half_cell_edge, 100.25 + half_cell_edge,
                   20.00 - half_cell_edge, 20.25 + half_cell_edge])
plt.colorbar(orientation='vertical')
plt.xticks(xnew)
plt.yticks(ynew)
plt.show()

"""
def test(lon, lat):
    if lon < 0 or lat < 0:
        print('error')

    lon_frac_part, lon_inte_part = math.modf(lon)
    lat_frac_part, lat_inte_part = math.modf(lat)

    era5_frac_parts = np.arange(0.0, 1.01, 0.25)

    for start_idx in range(len(era5_frac_parts) - 1):
        start = era5_frac_parts[start_idx]
        end = era5_frac_parts[start_idx + 1]

        if lon_frac_part >= start and lon_frac_part < end:
            lon1 = start + lon_inte_part
            lon2 = end + lon_inte_part

        if lat_frac_part >= start and lat_frac_part < end:
            lat1 = start + lat_inte_part
            lat2 = end + lat_inte_part

    try:
        return lat1, lat2, lon1, lon2
    except NameError:
        print((f"Fail getting ERA5 corners of SCS grid cell"))

print(test(114.35, 13.95))
"""
