# !/usr/bin/env python

import math

import numpy as np
import matplotlib.pyplot as plt

def convert_10_1(wspd, height):
    """Convert the wind speed at the the height of anemometer to
    the wind speed at the height of 10 meters by means of Xiaoping
    Xie et al.

    Parameters
    ----------
    wspd : float
        Wind speed at the height of anemometer.
    height : float
        The height of anemometer.

    Returns
    -------
    con_wspd : float
        Wind speed at the height of 10 meters.

    References
    ----------
    Xiaoping Xie, Jiansu Wei, and Liang Huang, Evaluation of ASCAT
    Coastal Wind Product Using Nearshore Buoy Data, Journal of Applied
    Meteorological Science 25 (2014), no. 4, 445–453.

    """
    if wspd <= 7:
        z0 = 0.0023
    else:
        z0 = 0.022
    kz = math.log(10/z0) / math.log(height/z0)
    con_wspd = wspd * kz

    return con_wspd

def convert_10_2(wspd, height):
    """Convert the wind speed at the the height of anemometer to
    the wind speed at the height of 10 meters by means of S.A HSU
    et al.

    Parameters
    ----------
    wspd : float
        Wind speed at the height of anemometer.
    height : float
        The height of anemometer.

    Returns
    -------
    con_wspd : float
        Wind speed at the height of 10 meters.

    References
    ----------
    Hsu, S. A., Eric A. Meindl, and David B. Gilhousen. "Determining
    the Power-Law Wind-Profile Exponent under Near-Neutral Stability
    Conditions at Sea." Journal of Applied Meteorology 33.6
    (1994): 757-765.

    """
    u1 = wspd
    z1 = height
    z2 = 10
    p = 0.11

    u2 = math.pow(z2/z1, p) * u1

    return u2

def test():
    h = 5
    spd = np.arange(0., 37.5, 0.5)
    spd_1 = []
    spd_2 = []
    spd_3 = []
    for s in spd:
        spd_1.append(convert_10_1(s, h))
        spd_2.append(convert_10_2(s, h))
        spd_3.append(convert_10_1(s, h) - convert_10_2(s, h))
    line_1, = plt.plot(spd, spd_1, 'r--', label = 'Xiaoping Xie')
    line_2, = plt.plot(spd, spd_2, 'g--', label = 'S.A. HSU')
    line_3, = plt.plot(spd, spd_3, 'b--', label = 'Difference')
    plt.legend(handles=[line_1, line_2, line_3])
    plt.xlabel('Wind Speed at %sm (m/s)' % h)
    plt.ylabel('Wind Speed at 10m (m/s)')
    plt.savefig('Compare two 10m wind speed coverters.png')
    plt.show()

if __name__ == '__main__':
    test()
