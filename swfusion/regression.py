import logging
import os
import shutil
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
from keras import backend as K
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
from keras.models import model_from_yaml
import yaml

import utils
import era5

Base = declarative_base()

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class Regression(object):

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

        self.smap_columns = ['x', 'y', 'lon', 'lat', 'windspd']
        # self.era5_columns = self.CONFIG['era5']['all_vars']

        self.era5_useless_columns = [
            'key', 'x_y'
        ]
        self.edge = self.CONFIG['era5']['subset_edge_in_degree']
        self.spa_resolu = self.CONFIG['era5']['spatial_resolution']
        self.lat_grid_points = [y * self.spa_resolu - 90 for y in range(
            self.CONFIG['era5']['lat_grid_points_number'])]
        self.lon_grid_points = [x * self.spa_resolu for x in range(
            self.CONFIG['era5']['lon_grid_points_number'])]

        self.epochs = 500
        self.batch_size = 32
        self.validation_split = 0.2

        self.predict_table_name_prefix = 'predicted_smap_tc'

        self.compare_zorders = self.CONFIG['plot']['zorders']['compare']

        self.wind_radii = self.CONFIG['wind_radii']

        utils.setup_database(self, Base)

        self.read_smap_era5(None, 15, 'small')
        self.bulid_DNN()
        self.train_DNN()
        self.predict_and_evaluate()

    def predict_and_evaluate(self):
        self.logger.info((f"""Predicting and evaluating"""))
        # Get ERA file in test period according to IBTrACS data in
        # SCS/WP ?

        # Generate ERA5 fields from ERA5 file

        # Predict SMAP windspd
        score = self.NN_model.evaluate(self.test, self.test_target,
                                       verbose=0)

        self.logger.info((f"""Evaluating on test dataset"""))
        metrics_names = self.NN_model.metrics_names
        for i in range(len(metrics_names)):
            print(f'{metrics_names[i]}: {score[i]}')

    def predict_old(self):
        the_mode = 'surface_all_vars'
        # Read ERA5 dataframe
        era5_ = era5.ERA5Manager(self.CONFIG, self.test_period,
                                 self.region, self.db_root_passwd,
                                 work=False, save_disk=self.save_disk)
        era5_table_names = era5_.get_era5_table_names(
            mode=the_mode)

        for table_name in era5_table_names:
            df = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
            df.drop(self.era5_useless_columns, axis=1, inplace=True)
            # Predict SMAP windspd
            smap_windspd_pre = self.NN_model.predict(df)

            table_name_suffix = table_name.split(the_mode)[1]
            predict_table_name = (f'{predict_table_name_prefix}'
                                  + f'{table_name_suffix}')

            pre_df = df.drop(self.era5_columns, axis=1)
            pre_df['windspd'] = smap_windspd_pre

            pre_df.to_sql(name=predict_table_name,
                          con=self.engine, if_exists = 'replace',
                          index=False)

    def compare_with_ibtracs(self):
        # Traverse all ibtracs record within test_period
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        for row in self.session.query(TCTable).filter(
            TCTable.date_time >= self.test_period[0],
            TCTable.date_time <= self.test_period[1]).yield_per(
            self.CONFIG['database']['batch_size']['query']):
            # Read corresponding predicted SMAP windspd dataframe
            dt_str = dt.strftime('%Y_%m%d_%H%M')
            table_name_suffix = f'_{row.sid}_{dt_str}'
            table_name = (f'{self.predict_table_name_prefix}'
                          + f'{table_name_suffix}')
            pre_df = pd.read_sql(f'SELECT * FROM {table_name}',
                                 self.engine)
            # Compare it with ibtracs
            self.compare_ibtracs_era5(row, pre_df, draw=True,
                                      draw_map=True, draw_bar=False)

    def compare_ibtracs_era5(self, ibtracs_table_row, pre_df,
                             draw, draw_map, draw_bar):
        tc_row = ibtracs_table_row
        lon_converted = tc_row.lon + 360 if tc_row.lon < 0 else tc_row.lon
        tc_radii = utils.get_radii_from_tc_row(tc_row)

        lats = sorted(list(set(pre_df['lat'])))
        lons = sorted(list(set(pre_df['lon'])))
        lon1, lon2 = lons.min(), lons.max()
        lat1, lat2 = lats.min(), lats.max()

        fig = plt.figure()
        if draw_map and draw_bar:
            fig.set_size_inches(25, 10)
            map_subplot = 121
            bar_subplot = 122
        else:
            fig.set_size_inches(10, 10)
            map_subplot = bar_subplot = 111

        # Draw ax1 to compare IBTrACS wind radii with ERA5 wind speed contour
        ax1 = fig.add_subplot(map_subplot, aspect='equal')
        ax1.axis([lon1, lon2, lat1, lat2])

        utils.draw_compare_basemap(ax1, lon1, lon2, lat1, lat2,
                                   self.compare_zorders)
        utils.set_basemap_title(ax1, tc_row, 'SMAP')

        center = (lon_converted, tc_row.lat)
        windspd = self._get_compare_smap_windspd_in_knots(
            pre_df, lats, lons)
        utils.draw_windspd(ax1, lats, lons, windspd,
                            self.compare_zorders)

        ibtracs_area = utils.draw_ibtracs_radii(ax1, center, tc_radii,
                                                self.wind_radii,
                                                self.compare_zorders)
        smap_area = utils.get_area_within_radii(ax1, lats, lons, windspd,
                                                self.wind_radii)

        utils.write_area_compare(self, tc_row, ibtracs_area,
                                 'smap', smap_area)

        if not draw:
            plt.close(fig)
            return

        if not draw_map:
            ax1.remove()

        # Draw ax2 to compare area within IBTrACS wind radii with
        # corresponding area of ERA5 wind speed contour
        ax2 = fig.add_subplot(bar_subplot)
        utils.draw_compare_area_bar(ax2, ibtracs_area, smap_area,
                                    'SMAP', tc_row)

        if not draw_bar:
            ax2.remove()

        fig.tight_layout()
        fig_dir = self.CONFIG["result"]["dirs"]["fig"]\
                ['ibtracs_vs_smap']
        fig_path = (f'{fig_dir}'
                    + f'smap_vs_ibtracs_{tc_row.sid}_'
                    + f'{tc_row.name}_{tc_row.date_time}_'
                    + f'{lon_converted}_{tc_row.lat}.png')
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    def _get_compare_smap_windspd_in_knots(self, pre_df, lats, lons):
        windspd = np.ndarray(shape=(len(lats), len(lons)),
                             dtype=float)
        for pt in pre_df:
            lon_idx = lons.index(pt.lon)
            lat_idx = lats.index(pt.lat)

            windspd[lat_idx][lon_idx] = 1.94384 * pt.windspd

        return windspd

    def train_DNN(self, skip_fit=False):
        self.logger.info((f"""Training DNN"""))
        if not skip_fit:
            self.NN_model.fit(self.train, self.target,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              validation_split=self.validation_split,
                              callbacks=self.callbacks_list, verbose=0)

        # Load weights file of the best model:
        # choose the best checkpoint
        weights_file = utils.find_best_NN_weights_file(
            self.checkpoint_dir)
        # load it
        self.NN_model.load_weights(weights_file)
        self.NN_model.compile(loss=root_mean_squared_error,
                              optimizer='adam',
                              metrics=[root_mean_squared_error])

        # self.logger.info((f"""Evaluating on train dataset"""))
        # score = self.NN_model.evaluate(self.train, self.target,
        #                                verbose=0)

        # metrics_names = self.NN_model.metrics_names
        # for i in range(len(metrics_names)):
        #     print(f'{metrics_names[i]}: {score[i]}')

        # evaluation_dir = self.CONFIG['regression']['dirs']['tc']\
        #         ['evaluation']

    def bulid_DNN(self):
        self.logger.info((f"""Building DNN"""))
        self.NN_model = Sequential()
        self.NN_model.add(Dense(128, kernel_initializer='normal',
                                input_dim = self.train.shape[1],
                                activation='relu'))
        for i in range(3):
            self.NN_model.add(Dense(256, kernel_initializer='normal',
                                    activation='relu'))
        self.NN_model.add(Dense(1, kernel_initializer='normal',
                                activation='linear'))
        self.NN_model.compile(loss=root_mean_squared_error,
                              optimizer='adam',
                              metrics=[root_mean_squared_error])

        self.NN_model.summary()

        yaml_string = self.NN_model.to_yaml()
        # model = model_from_yaml(yaml_string)
        model_summary_dir = self.CONFIG['regression']['dirs']['tc']\
                ['model_summary']
        os.makedirs(model_summary_dir, exist_ok=True)
        model_summary_path = f'{model_summary_dir}model.yml'
        with open(model_summary_path, 'w') as outfile:
            yaml.dump(yaml_string, outfile, default_flow_style=False)

        checkpoint_root_dir = self.CONFIG['regression']['dirs']\
                ['tc']['checkpoint']

        self.checkpoint_dir = (f"""{checkpoint_root_dir}"""
                               f"""{self.condition}/""")
        # Remove old checkpoint files
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        # Create a new checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint_path = (f"""{self.checkpoint_dir}"""
                           f"""{checkpoint_name}""")

        checkpoint = ModelCheckpoint(checkpoint_path,
                                     monitor='val_loss',
                                     verbose=2, save_best_only=True,
                                     mode='auto')
        self.callbacks_list = [checkpoint]

    def read_smap_era5(self, col_name, divide, large_or_small):
        self.logger.info((f"""Reading SMAP-ERA5 data"""))
        combined, target, test_target, train_length = \
                self.get_combined_data(col_name, divide, large_or_small)

        # num_cols = utils.get_dataframe_cols_with_no_nans(combined,
        #                                                  'num')
        # print ('Number of numerical columns with no nan values :',
        #        len(num_cols))

        self.train, self.test = self.split_combined(combined,
                                                    train_length)
        self.target = target
        self.test_target = test_target

        # train_data, test_data = self._get_train_test()
        # train_data = self.train
        # train_data['Target'] = target
        # self._show_correlation_heatmap(train_data)
        # breakpoint()

    def split_combined(self, combined, train_length):
        train = combined[:train_length]
        test = combined[train_length:]

        return train, test

    def get_combined_data(self, col_name, divide, large_or_small):
        train, test = self._get_train_test(col_name, divide,
                                           large_or_small)
        train_length = len(train)

        target_name = self.CONFIG['regression']['target']['smap_era5']
        target = getattr(train, target_name)
        test_target = getattr(test, target_name)
        train.drop([target_name], axis=1, inplace=True)
        test.drop([target_name], axis=1, inplace=True)

        combined = train.append(test)
        combined.reset_index(inplace=True)
        combined.drop(['index'], axis=1, inplace=True)

        return combined, target, test_target, train_length

    def _get_train_test(self, col_name, divide, large_or_small):
        # years = [y for y in range(self.train_period[0].year,
        #                           self.train_period[1].year+1)]
        years = [2015, 2016, 2017, 2018, 2019]
        df = None
        for y in years:
            table_name = f'tc_smap_era5_{y}'
            if (utils.get_class_by_tablename(self.engine, table_name)
                is not None):
                tmp_df = pd.read_sql(f'SELECT * FROM {table_name}',
                                     self.engine)
                if df is None:
                    df = tmp_df
                else:
                    df = df.append(tmp_df)

        df.drop(self.CONFIG['regression']['useless_columns']\
                ['smap_era5'], axis=1, inplace=True)

        if col_name is not None:
            df, self.condition = \
                    utils.filter_dataframe_by_column_value_divide(
                        df, col_name, divide, large_or_small)
        else:
            self.condition = 'all'

        train, test = train_test_split(df, test_size=0.2)

        return train, test

    def _show_dataframe_hist(self, df, columns):
        df.hist(column=columns, figsize = (12,10))
        plt.show()

    def _show_correlation_heatmap(self, data):
        C_mat = data.corr()
        fig = plt.figure(figsize=(15, 15))
        sb.heatmap(C_mat, vmax=.8, square=True)
        plt.show()
