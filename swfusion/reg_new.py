import datetime
import logging
import math
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
from mpl_toolkits.basemap import Basemap
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils
import era5

Base = declarative_base()

class NewReg(object):

    def __init__(self, CONFIG, train_period, test_period, region, passwd,
                 save_disk):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.train_period = train_period
        self.test_period = test_period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.save_disk = save_disk

        self.satel_names = self.CONFIG['satel_names']
        self.lat1, self.lat2 = region[0], region[1]
        self.lon1, self.lon2 = region[2], region[3]

        self.useless_columns = self.CONFIG['regression']\
                ['useless_columns']
        self.edge = self.CONFIG['era5']['subset_edge_in_degree']
        self.spa_resolu = self.CONFIG['era5']['spatial_resolution']
        self.lat_grid_points = [y * self.spa_resolu - 90 for y in range(
            self.CONFIG['era5']['lat_grid_points_number'])]
        self.lon_grid_points = [x * self.spa_resolu for x in range(
            self.CONFIG['era5']['lon_grid_points_number'])]

        self.epochs = 500
        self.batch_size = 32
        self.validation_split = 0.2

        self.grid_lons = None
        self.grid_lats = None
        self.grid_x = None
        self.grid_y = None
        # Load 4 variables above
        utils.load_grid_lonlat_xy(self)

        self.predict_table_name_prefix = 'predicted_smap_tc'

        self.wind_radii = self.CONFIG['wind_radii']

        utils.setup_database(self, Base)
        self.months = self._gen_months_in_period()
        self.zorders = self.CONFIG['plot']['zorders']['scs_basemap']

        for satel_name in self.satel_names:
            self.logger.info((f"""Training regression model of """
                              f"""{satel_name} and predicting data """
                              f"""using ERA5 reanalysis"""))
            self.read_satel_era5(satel_name)
            self.make_satel_DNNs(satel_name)
            self.train_satel_DNNs(satel_name)
            self.predict(satel_name)

    def predict(self, satel_name):
        scs_era5_table_names, scs_era5_table_datetime = \
                self.gen_scs_era5_table_names()

        for table_name, table_datetime in zip(
            scs_era5_table_names, scs_era5_table_datetime):

            # Load ERA5 table
            df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
            # Drop useless columns
            df.drop(
                self.CONFIG['regression']['useless_columns']['era5'],
                axis=1, inplace=True
            )
            # Predict uv wind component of satellite
            u_v_wind_pre = dict()
            for idx, col_name in enumerate(self.target.keys()):
                if not idx:
                    u_v_wind_pre['u'] = \
                            self.NN_model[col_name].predict(df)
                else:
                    u_v_wind_pre['v'] = \
                            self.NN_model[col_name].predict(df)

            pts_num = len(df.index)
            windspd_pre = []
            for i in range(pts_num):
                windspd_pre.append(
                    math.sqrt(
                        (u_v_wind_pre['u'][i] ** 2
                         + u_v_wind_pre['v'][i] ** 2)
                    )
                )
            del u_v_wind_pre

            # Display prediction on map
            self.draw_prediction(table_datetime, satel_name, df,
                                 windspd_pre)

    def draw_prediction(self, table_datetime, satel_name, df,
                        windspd_pre_list):

        windspd_pre_matrix = np.zeros(shape=(len(self.grid_lats),
                                             len(self.grid_lons)),
                                      dtype=float)
        windspd_ecmwf_matrix = np.zeros(shape=(len(self.grid_lats),
                                               len(self.grid_lons)),
                                        dtype=float)

        rows_num = len(df.index)
        grid_lats_num = len(self.grid_lats)
        grid_lons_num = len(self.grid_lons)

        for i in range(rows_num):
            row = df.iloc[i]
            lat_idx = self.grid_lats.index(row.lat)
            lon_idx = self.grid_lons.index(row.lon)
            # Only for display wind cell according to satellite's
            # spatial resolution
            for y_offset in range(-2, 3):
                sub_lat_idx = lat_idx + y_offset
                if sub_lat_idx < 0 or sub_lat_idx >= grid_lats_num:
                    continue

                for x_offset in range(-2, 3):
                    sub_lon_idx = lon_idx + x_offset
                    if sub_lon_idx < 0 or sub_lon_idx >= grid_lons_num:
                        continue

                    windspd_pre_matrix[sub_lat_idx][sub_lon_idx] = \
                            windspd_pre_list[i]
                    windspd_ecmwf_matrix[sub_lat_idx][sub_lon_idx] = \
                            math.sqrt(
                                (row.u_component_of_wind ** 2
                                 + row.v_component_of_wind ** 2)
                            )

        fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharey=True)

        reanaly_and_satel = [windspd_ecmwf_matrix, windspd_pre_matrix]
        max_windspd = max(windspd_ecmwf_matrix.max(),
                          windspd_pre_matrix.max())
        min_windspd = min(windspd_ecmwf_matrix.min(),
                          windspd_pre_matrix.min())
        for idx, wind in enumerate(reanaly_and_satel):
            ax = axs[idx]
            ax.axis([self.lon1, self.lon2, self.lat1, self.lat2])
            utils.draw_SCS_basemap(self, ax)

            if not idx:
                ax.set_title(f'ECMWF ERA5')
            else:
                ax.set_title(f'Prediction of {satel_name} observation')

            self.draw_reg_windspd(fig, ax, self.grid_lons, self.grid_lats,
                                  wind, min_windspd, max_windspd)

        fig_name = f'regression_{table_datetime}_{satel_name}.png'
        fig_dir = self.CONFIG['result']['dirs']['fig']
        plt.savefig(f'{fig_dir}{fig_name}')
        plt.clf()

    def draw_reg_windspd(self, fig, ax, lons, lats, windspd,
                            min_windspd, max_windspd):
        X, Y = np.meshgrid(lons, lats)
        Z = windspd

        # windspd_levels = [5*x for x in range(1, 15)]
        windspd_levels = np.linspace(0.01, max_windspd, 10)

        # cs = ax.contour(X, Y, Z, levels=windspd_levels,
        #                 zorder=self.zorders['contour'], colors='k')
        # ax.clabel(cs, inline=1, colors='k', fontsize=10)
        cf = ax.contourf(X, Y, Z, levels=windspd_levels,
                         zorder=self.zorders['contourf'],
                         cmap=plt.cm.rainbow)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(cf, cax=cax, orientation='vertical', format='%.1f')

    def gen_scs_era5_table_names(self):
        table_names = []
        table_datetime = []

        delta = self.test_period[1] - self.test_period[0]
        months = set()

        if delta.seconds > 0:
            days_num = delta.days + 1
        else:
            days_num = delta.days

        for day in range(days_num):
            dt_cursor = (self.test_period[0]
                         + datetime.timedelta(days=day))
            for hourtime in range(0, 2400,
                                  self.CONFIG['product']\
                                  ['temporal_resolution']):
                table_name = utils.gen_scs_era5_table_name(
                    dt_cursor, hourtime)
                table_names.append(table_name)

                table_datetime.append(dt_cursor.replace(
                    second=0, microsecond=0, minute=0,
                    hour=int(hourtime/100)))

        return table_names, table_datetime

    def train_satel_DNNs(self, satel_name, skip_fit=False):
        for col_name in self.target.keys():
            self.train_single_DNN(satel_name, col_name, skip_fit)

    def train_single_DNN(self, satel_name, col_name, skip_fit):
        if not skip_fit:
            self.NN_model[col_name].fit(
                self.train, self.target[col_name], epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=self.callbacks_list[col_name],
                verbose=0
            )

        # Load weights file of the best model:
        # choose the best checkpoint
        weights_file = self.find_best_weights_file(col_name)
        # load it
        self.NN_model[col_name].load_weights(weights_file)
        self.NN_model[col_name].compile(loss='mean_squared_error',
                                        optimizer='adam',
                                        metrics=['mean_squared_error'])
        score = self.NN_model[col_name].evaluate(self.train,
                                                 self.target[col_name],
                                                 verbose=0)
        metrics_names = self.NN_model[col_name].metrics_names

        for i in range(len(metrics_names)):
            print(f'{metrics_names[i]}: {score[i]}')

    def find_best_weights_file(self, col_name):
        file_names = [f for f in os.listdir(self.checkpoint_dir[col_name])
                      if f.endswith('.hdf5')]
        max_epoch = -1
        best_weights_file_name = None

        for file in file_names:
            epoch = int(file.split('-')[1])
            if epoch > max_epoch:
                max_epoch = epoch
                best_weights_file_name = file

        return f'{self.checkpoint_dir[col_name]}{best_weights_file_name}'

    def make_satel_DNNs(self, satel_name):
        self.NN_model = dict()
        self.checkpoint_dir = dict()
        self.callbacks_list = dict()

        for col_name in self.target.keys():
            self.make_single_DNN(satel_name, col_name)

    def make_single_DNN(self, satel_name, col_name):
        self.NN_model[col_name] = Sequential()

        self.NN_model[col_name].add(
            Dense(128, kernel_initializer='normal',
                  input_dim = self.train.shape[1],
                  activation='relu'))

        for i in range(3):
            self.NN_model[col_name].add(
                Dense(256, kernel_initializer='normal',
                      activation='relu'))

        self.NN_model[col_name].add(
            Dense(1, kernel_initializer='normal', activation='linear'))

        self.NN_model[col_name].compile(loss='mean_squared_error',
                                        optimizer='adam',
                                        metrics=['mean_squared_error'])

        self.NN_model[col_name].summary()

        self.checkpoint_dir[col_name] = (
            f"""{self.CONFIG['regression']['dirs']['checkpoint']}"""
            f"""{satel_name}/"""
            f"""{col_name}/"""
        )
        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint_path = (f"""{self.checkpoint_dir[col_name]}"""
                           f"""{checkpoint_name}""")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = ModelCheckpoint(checkpoint_path,
                                     monitor='val_loss',
                                     verbose=0, save_best_only=True,
                                     mode ='auto')
        self.callbacks_list[col_name] = [checkpoint]

    def read_satel_era5(self, satel_name):
        combined, target, train_length = self.get_combined_data(
            satel_name)

        num_cols = utils.get_dataframe_cols_with_no_nans(combined, 'num')
        # print ('Number of numerical columns with no nan values :',
        #        len(num_cols))
        # self.show_heatmap(satel_name, target)
        self.train, self.test = self.split_combined(combined,
                                                    train_length)
        self.target = target

    def split_combined(self, combined, train_length):
        train = combined[:train_length]
        test = combined[train_length:]

        return train, test

    def get_combined_data(self, satel_name):
        train, test = self._get_train_test(satel_name)
        train_length = len(train)

        target = dict()
        for col_name in self.CONFIG['regression']['target'][satel_name]:
            target[col_name] = getattr(train, col_name)
            train.drop([col_name], axis=1, inplace=True)
            test.drop([col_name], axis=1, inplace=True)

        combined = train.append(test)
        combined.reset_index(inplace=True)
        combined.drop(['index'], axis=1, inplace=True)

        return combined, target, train_length

    def show_heatmap(self, satel_name, target):
        train_data, test_data = self._get_train_test(satel_name)
        train_data['Target'] = target

        C_mat = train_data.corr()
        fig = plt.figure(figsize=(15, 15))
        sb.heatmap(C_mat, vmax=.8, square=True)
        plt.show()

    def _get_train_test(self, satel_name):
        all_train = []
        all_test = []

        for dt in self.months:
            table_name = (f"""{satel_name}_{dt.year}"""
                          f"""_{str(dt.month).zfill(2)}""")
            df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
            df.drop(self.useless_columns[satel_name], axis=1,
                    inplace=True)
            train, test = train_test_split(df, test_size=0.2)
            # Check the type of train and test, then consider
            # concate them 
            all_train.append(train)
            all_test.append(test)

        all_train_df = pd.concat(all_train)
        all_test_df = pd.concat(all_test)

        return all_train_df, all_test_df

    def _gen_months_in_period(self):
        delta = self.train_period[1] - self.train_period[0]
        months = set()

        if delta.seconds > 0:
            days_num = delta.days + 1
        else:
            days_num = delta.days

        for day in range(days_num):
            dt_cursor = (self.train_period[0]
                         + datetime.timedelta(days=day))
            dt_month = dt_cursor.replace(second=0, microsecond=0,
                                         minute=0, hour=0, day=2)
            months.add(dt_month)

        return months
