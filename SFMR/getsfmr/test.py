# !/usr/bin/env python

import pickle

import numpy as np

import read_SFMR

def test():
    pickle_path = '../SFMR_pickle/20150804U1_guillermo.pkl'
    with open(pickle_path, 'rb') as fr:
        hurr_pickle = pickle.load(fr)
    print(len(hurr_pickle['wspd']))
    for i in range(3):
        print(hurr_pickle['wspd'][i])
        print(hurr_pickle['precise_time'][i])

    nc_path = '../hurr_data/2015/guillermo/AFRC_SFMR20150804U1.nc'
    hurr_nc = read_SFMR.ReadNetcdf(nc_path)
    var = hurr_nc.dataset.variables
    print(len(var['SWS']))
    for idx, spd in enumerate(var['SWS']):
        if spd.mask == True:
            continue
        else:
            print(spd)
            print(idx)
            break

if __name__ == '__main__':
    test()
