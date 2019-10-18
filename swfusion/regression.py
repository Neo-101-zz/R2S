import logging
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from sqlalchemy.orm import mapper
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

import utils

Base = declarative_base()

class Regression(object):

    def __init__(self, CONFIG, period, region, passwd):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.period = period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None

        self.smap_columns = ['x', 'y', 'lon', 'lat', 'windspd']
        self.era5_columns = ['divergence', 'fraction_of_cloud_cover',
                             'geopotential', 'ozone_mass_mixing_ratio',
                             'potential_vorticity', 'relative_humidity',
                             'specific_cloud_ice_water_content',
                             'specific_cloud_liquid_water_content',
                             'specific_humidity',
                             'specific_rain_water_content',
                             'specific_snow_water_content', 'temperature',
                             'u_component_of_wind', 'v_component_of_wind',
                             'vertical_velocity', 'vorticity_relative']
        self.useless_columns = ['key', 'match_sid',
                                'temporal_window_mins',
                                'satel_tc_diff_mins', 'tc_datetime',
                                'satel_datetime', 'era5_datetime',
                                'satel_datetime_lon_lat',
                                'satel_era5_diff_mins']
        self.epochs = 500
        self.batch_size = 32
        self.validation_split = 0.2

        utils.setup_database(self, Base)

        self.read_era5_smap()
        self.make_DNN()
        self.train_DNN()

    def _get_era5_table_name(self, mode='surface_all_vars'):
        table_names = []
        # Get TC table and count its row number
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        # Loop all row of TC table
        for row in self.session.query(TCTable).filter(
            TCTable.date_time >= self.period[0],
            TCTable.date_time <= self.period[1]).yield_per(
            self.CONFIG['database']['batch_size']['query']):

            # Get TC datetime
            tc_datetime = row.date_time

            # Get hit result and range of ERA5 data matrix near
            # TC center
            hit, lat1, lat2, lon1, lon2 = \
                    utils.get_subset_range_of_grib(
                        row.lat, row.lon, self.lat_grid_points,
                        self.lon_grid_points, self.edge)
            if not hit:
                continue

            dirs = ['nw', 'sw', 'se', 'ne']
            r34 = dict()
            r34['nw'], r34['sw'], r34['se'], r34['ne'] = \
                    row.r34_nw, row.r34_sw, row.r34_se, row.r34_ne
            skip_compare = False
            for dir in dirs:
                if r34[dir] is None:
                    skip_compare = True
                    break
            if skip_compare:
                continue

            # Get name, sqlalchemy Table class and python original class
            # of ERA5 table
            table_name, sa_table, ERA5Table = self.get_era5_table_class(
                mode, row.sid, tc_datetime)

            table_names.append(table_name)

        return table_names

    def predict(self):
        # Read ERA5 dataframe
        era5_table_names = self._get_era5_table_name()
        for table_name in era5_table_names:
            df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
        # Predict SMAP windspd
        pass

    def train_DNN(self):
        # self.NN_model.fit(self.train, self.target, epochs=self.epochs,
        #                   batch_size=self.batch_size,
        #                   validation_split=self.validation_split,
        #                   callbacks=self.callbacks_list)

        # Load weights file of the best model:
        # choose the best checkpoint
        weights_file = 'Weights-438--1.16947.hdf5'
        # load it
        self.NN_model.load_weights(weights_file)
        self.NN_model.compile(loss='mean_absolute_error',
                              optimizer='adam',
                              metrics=['mean_absolute_error'])
        score = self.NN_model.evaluate(self.train, self.target,
                                       verbose=0)
        metrics_names = self.NN_model.metrics_names
        for i in range(len(metrics_names)):
            print(f'{metrics_names[i]}: {score[i]}')

    def make_DNN(self):
        self.NN_model = Sequential()
        self.NN_model.add(Dense(128, kernel_initializer='normal',
                                input_dim = self.train.shape[1],
                                activation='relu'))
        for i in range(3):
            self.NN_model.add(Dense(256, kernel_initializer='normal',
                                    activation='relu'))
        self.NN_model.add(Dense(1, kernel_initializer='normal',
                                activation='linear'))
        self.NN_model.compile(loss='mean_absolute_error',
                              optimizer='adam',
                              metrics=['mean_absolute_error'])

        self.NN_model.summary()

        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_name,
                                     monitor='val_loss',
                                     verbose=1, save_best_only=True,
                                     mode ='auto')
        self.callbacks_list = [checkpoint]

    def read_era5_smap(self):
        train_data, test_data = self._get_train_test()
        combined, target, train_length = self.get_combined_data()

        num_cols = self.get_cols_with_no_nans(combined, 'num')
        # print ('Number of numerical columns with no nan values :',
        #        len(num_cols))

        # train_data['Target'] = target

        # C_mat = train_data.corr()
        # fig = plt.figure(figsize=(15, 15))
        # sb.heatmap(C_mat, vmax=.8, square=True)
        # plt.show()

        self.train, self.test = self.split_combined(combined,
                                                    train_length)
        self.target = target

    def split_combined(self, combined, train_length):
        train = combined[:train_length]
        test = combined[train_length:]

        return train , test

    def get_combined_data(self):
        train, test = self._get_train_test()
        train_length = len(train)

        target = train.windspd
        train.drop(['windspd'], axis=1, inplace=True)
        test.drop(['windspd'], axis=1, inplace=True)

        combined = train.append(test)
        combined.reset_index(inplace=True)
        combined.drop(['index'], axis=1, inplace=True)

        return combined, target, train_length

    def _get_train_test(self):
        table_name = 'smap_2015'
        df = pd.read_sql('SELECT * FROM smap_2018', self.engine)
        df.drop(self.useless_columns, axis=1, inplace=True)
        train, test = train_test_split(df, test_size=0.2)

        return train, test

    def get_cols_with_no_nans(self, df, col_type):
        '''
        Arguments :
        df : The dataframe to process
        col_type : 
              num : to only get numerical columns with no nans
              no_num : to only get nun-numerical columns with no nans
              all : to get any columns with no nans    
        '''
        if (col_type == 'num'):
            predictors = df.select_dtypes(exclude=['object'])
        elif (col_type == 'no_num'):
            predictors = df.select_dtypes(include=['object'])
        elif (col_type == 'all'):
            predictors = df
        else :
            print('Error : choose a type (num, no_num, all)')
            return 0
        cols_with_no_nans = []
        for col in predictors.columns:
            if not df[col].isnull().any():
                cols_with_no_nans.append(col)

        return cols_with_no_nans

    def _show_dataframe(self, df):
        df.hist(column=self.era5_columns, figsize = (12,10))
        plt.show()

