import logging
import math
import os
import pickle
import shutil
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy.orm import mapper
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from mpl_toolkits.basemap import Basemap
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
from keras.models import model_from_yaml
import yaml
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

import utils
import era5

Base = declarative_base()

class DNNLayerConf(object):

    def __init__(self, units):
        self.units = units

class DNNLayerStructure(object):

    def __init__(self):
        self.input_layer = None
        self.hidden_layer = None
        self.output_layer = None

    def set_input_layer(self, units):
        self.input_layer = DNNLayerConf(units)

    def set_output_layer(self, units):
        self.output_layer = DNNLayerConf(units)

    def set_hidden_layer(self, hidden_list):
        self.hidden_layer = []
        for hidden in hidden_list:
            self.hidden_layer.append(DNNLayerConf(hidden['units']))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir,
                                                  **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items()
                    if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

class Regression(object):

    def __init__(self, CONFIG, train_period, test_period, region, basin,
                 passwd, save_disk, instructions):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.train_period = train_period
        self.test_period = test_period
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.save_disk = save_disk
        self.basin = basin
        self.instructions = instructions

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

        self.model_dir = dict()
        self.model_dir['xgb'] = self.CONFIG['regression']['dirs']['tc']\
                ['xgboost']['model']

        self.validation_split = 0.2

        self.predict_table_name_prefix = 'predicted_smap_tc'

        self.compare_zorders = self.CONFIG['plot']['zorders']['compare']

        self.wind_radii = self.CONFIG['wind_radii']

        utils.setup_database(self, Base)

        self.target_name = self.CONFIG['regression']['target']\
                ['smap_era5']

        self.read_smap_era5(False, '', 15, 'large')
        if 'dt' in self.instructions:
            self.decision_tree()
        if 'xgb' in self.instructions:
            # self.evaluation_df_visualition()
            self.xgb_regressor()
            # self.xgb_importance()
        if 'dnn' in self.instructions:
            self.deep_neural_network()

    def evaluation_df_visualition(self):
        reg_tc_dir = self.CONFIG['regression']['dirs']['tc']
        if self.reg_xgb:
            df_dir = reg_tc_dir['xgboost']['evaluation']
            df_files = [f for f in os.listdir(df_dir)
                        if f.endswith('.pkl')]
            for file in df_files:
                fig_name_prefix = file.split('.pkl')[0]
                df = pd.read_pickle(f'{df_dir}{file}')
                x_is_learning_rate = df.pivot(index='learning_rate',
                                              columns='max_depth',
                                              values='mse')
                x_is_learning_rate.plot()
                plt.ylabel('mse')
                plt.savefig((f"""{df_dir}{fig_name_prefix}_"""
                             f"""mse_learning_rate.png"""))
                plt.clf()

                x_is_max_depth = df.pivot(index='max_depth',
                                          columns='learning_rate',
                                          values='mse')
                x_is_max_depth.plot()
                plt.ylabel('mse')
                plt.savefig((f"""{df_dir}{fig_name_prefix}_"""
                             f"""mse_max_depth.png"""))
                plt.clf()

    def xgb_importance(self):
        model_files = [f for f in os.listdir(self.model_dir['xgb'])
                       if f.endswith('.pickle.dat')]
        min_mse = 99999999
        best_model_name = None
        # Find best model
        for file in model_files:
            mse = float(file.split('.pickle')[0].split('mse_')[1])
            if mse < min_mse:
                min_mse = mse
                best_model_name = file
        # load model from file
        best_model = pickle.load(
            open(f'{self.model_dir["xgb"]}{best_model_name}', 'rb'))
        fig_dir = self.CONFIG['regression']['dirs']['tc']['xgboost']\
                ['importance']
        os.makedirs(fig_dir, exist_ok=True)
        selected_features_num = int(self.train.shape[1] / 3)

        for max_feat_num in [None, selected_features_num]:
            if max_feat_num is None:
                fig = plt.figure(figsize=(20, 40))
            else:
                fig = plt.figure(figsize=(20, 20))
            for imp_type in ['weight', 'gain', 'cover']:
                ax = xgb.plot_importance(
                    best_model, importance_type=imp_type,
                    max_num_features=max_feat_num,
                    title=f'Feature importance ({imp_type})')

                plt.subplots_adjust(left=0.4, right=0.9,
                                    top=0.9, bottom=0.1)
                for text in ax.texts:
                    text.set_text(text._text.split('.')[0])

                if max_feat_num is None:
                    plt.tick_params('y', labelsize=4)
                    for text in ax.texts:
                        text.set_fontsize(4)
                    fig_name = f'{imp_type}_all_features.png'
                else:
                    plt.tick_params('y', labelsize=6)
                    for text in ax.texts:
                        text.set_fontsize(6)
                    fig_name = (f"""{imp_type}_top_{max_feat_num}"""
                                f"""_features.png""")
                plt.tight_layout()

                plt.savefig(f'{fig_dir}{fig_name}', dpi=300)
                plt.clf()

    def xgb_regressor(self):
        self.logger.info((f"""Regressing with XGBRegressor"""))

        # 5 ~ 10, best: 10
        depth_range = [d + 5 for d in range(6)]
        # 0 ~ 0.2, best: 0.1
        learning_rate_range = [0.05 * r for r in range(1, 5)]
        n_estimators_range = [200 * i for i in range(1, 6)]
        total = len(depth_range ) * len(learning_rate_range)
        count = 0

        best_max_depth = 10
        best_learning_rate = 0.1
        # actually is 1000, 100 for test
        best_estimators_num = 1000
        # total = len(n_estimators_range)

        column_names = ['estimators_num', 'mse']
        all_tests = []

        os.makedirs(self.model_dir['xgb'], exist_ok=True)

        # for depth in depth_range:
        #     for learning_rate in learning_rate_range:
        # for estimators_num in n_estimators_range:
        #     count += 1
        #     print(f'\r{count}/{total}', end='')

        # one_test = dict()
        # one_test['estimators_num'] = estimators_num

        model = xgb.XGBRegressor(max_depth=best_max_depth,
                                 learning_rate=best_learning_rate,
                                 n_estimators=best_estimators_num,
                                 objective='reg:squarederror')
        model.fit(self.train, self.target)

        self.logger.info((f"""Evaluating on test set"""))
        predicted = model.predict(data=self.test)
        truth = self.test_target.to_numpy()
        mse = mean_squared_error(truth, predicted)
        print(f'MSE: {mse}')
        # one_test['mse'] = mse

        # save model to file
        pickle.dump(model, open((f"""{self.model_dir["xgb"]}"""
                                 f"""{self.basin}_mse_{mse}.pickle.dat"""),
                                'wb'))

        # all_tests.append(one_test)

        utils.delete_last_lines()
        print('Done')

        # df = pd.DataFrame(all_tests, columns=column_names)
        # evaluation_dir = self.CONFIG['regression']['dirs']['tc']\
        #         ['xgboost']['evaluation']
        # os.makedirs(evaluation_dir, exist_ok=True)
        # df.to_pickle(f'{evaluation_dir}test_set.pkl')
        # print(df)

    def decision_tree(self):
        self.logger.info((f"""Regressing with decision tree"""))
        self.logger.info((f"""Evaluating on test set"""))

        for depth in range(3, 20):
            tree = DecisionTreeRegressor(criterion='mse', max_depth=depth)
            tree.fit(self.train, self.target)

            predicted = tree.predict(self.test)
            truth = self.test_target.to_numpy()
            mse = mean_squared_error(truth, predicted)

            print(f"""Max depth {depth}: MSE {mse:.2f}""")

    def deep_neural_network(self):
        self.logger.info((f"""Regressing with deep neural network"""))
        """
        At least 4 variables to trail and error:

        Number of hidden layers
        -----------------------
        3, 6, 9

        Number of neurons
        -----------------
        Plan A: 1/3, 2/3, 3/3 of the size of input layer, plus the size of
        output layer.
        Plan B: 64, 128, 256

        The number of hidden neurons should be between the size of the
        input layer and the size of the output layer.
        The number of hidden neurons should be 2/3 the size of the
        input layer, plus the size of the output layer.
        The number of hidden neurons should be less than twice the
        size of the input layer.

        Learning rate
        -------------
        First test on 0.0001, 0.001 and 0.01.  Then ajust the range more
        precisely.

        Batch size
        ----------
        2, 4, 8, 16, 32

        Epochs
        ------
        100, 500, 1000

        Optimizer
        ---------
        Adam

        """
        # Numbers of hidden layer
        hidden_layers_num_range = [3 * i for i in range(1, 4)]
        # hidden_neurons_num_range = [
        #     int(self.train.shape[1] * x / 3) for x in range(1, 4)
        # ]
        hidden_neurons_num_range = [32, 64, 128, 256]
        # Learning rate
        self.learning_rate = 0.001
        # Batch size
        self.batch_size = 32
        # Epochs
        self.epochs = 500

        column_names = ['hidden_layers_num', 'hidden_neurons_num',
                        'mse']
        all_tests = []

        for hidden_layers_num in hidden_layers_num_range:
            for hidden_neurons_num in hidden_neurons_num_range:
                # Structure of neural network
                self.DNN_structure = self.gen_DNN_structure(
                    hidden_neurons_num, hidden_layers_num)
                self.DNN_structure_str = self.gen_DNN_structure_str()

                self.build_DNN()
                self.train_DNN()

                self.logger.info((f"""Evaluating on test set"""))
                # Predict SMAP windspd
                score = self.NN_model.evaluate(self.test,
                                               self.test_target,
                                               verbose=0)
                metrics_names = self.NN_model.metrics_names
                one_test = dict()
                one_test['hidden_layers_num'] = hidden_layers_num
                one_test['hidden_neurons_num'] = hidden_neurons_num
                one_test['mse'] = score[
                    metrics_names.index('mean_squared_error')
                ]
                print(one_test)

                all_tests.append(one_test)

        df = pd.DataFrame(all_tests, columns=column_names)
        evaluation_dir = self.CONFIG['regression']['dirs']['tc']\
                ['neural_network']['evaluation']
        os.makedirs(evaluation_dir, exist_ok=True)
        df.to_pickle(f'{evaluation_dir}test_set.pkl')
        print(df)

    def gen_DNN_structure(self, hidden_neurons_num, hidden_layers_num):
        structure = DNNLayerStructure()
        structure.set_input_layer(hidden_neurons_num)
        hidden_list = []
        for i in range(hidden_layers_num):
            hidden_list.append({'units': hidden_neurons_num})
        structure.set_hidden_layer(hidden_list)
        structure.set_output_layer(1)

        return structure

    def gen_DNN_structure_str(self):
        structure_str = (
            f"""input_{self.DNN_structure.input_layer.units}"""
            f"""_hidden"""
        )
        for hidden in self.DNN_structure.hidden_layer:
            structure_str = (f"""{structure_str}"""
                               f"""_{hidden.units}""")
        structure_str = (
            f"""{structure_str}_output_"""
            f"""{self.DNN_structure.output_layer.units}""")

        return structure_str

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
        strucuture_str = ''
        self.logger.info((f"""Training DNN"""))
        if not skip_fit:
            history = self.NN_model.fit(
                self.train, self.target, epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=self.callbacks_list, verbose=0)

            fig_dir = self.CONFIG['regression']['dirs']['tc']\
                    ['neural_network']['fit_history']
            os.makedirs(fig_dir, exist_ok=True)

            fig_name_suffix = (f"""{self.DNN_structure_str}.png""")

            # Summarize history for mean squared error
            plt.plot(history.history['mean_squared_error'])
            plt.plot(history.history['val_mean_squared_error'])
            plt.title('model mean_squared_error')
            plt.ylabel('mean_squared_error')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig((f"""{fig_dir}mean_squared_error_"""
                         f"""{fig_name_suffix}"""))
            plt.clf()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(f'{fig_dir}loss_{fig_name_suffix}')
            plt.clf()

        # Load weights file of the best model:
        # choose the best checkpoint
        weights_file = utils.find_best_NN_weights_file(
            self.checkpoint_dir)
        # load it
        self.NN_model.load_weights(weights_file)
        adam = optimizers.Adam(lr=self.learning_rate)
        self.NN_model.compile(loss='mean_squared_error',
                              optimizer=adam,
                              metrics=['mean_squared_error'])

    def build_DNN(self):
        self.logger.info((f"""Building DNN"""))
        input_confs = self.DNN_structure.input_layer
        hidden_confs = self.DNN_structure.hidden_layer
        output_confs = self.DNN_structure.output_layer

        K.clear_session()
        self.NN_model = Sequential()
        # Input layer
        self.NN_model.add(Dense(input_confs.units,
                                kernel_initializer='normal',
                                input_dim=self.train.shape[1],
                                activation='relu'))
        # Hidden layers
        for hidden in hidden_confs:
            self.NN_model.add(Dense(hidden.units,
                                    kernel_initializer='normal',
                                    activation='relu'))
        # Output layer
        self.NN_model.add(Dense(output_confs.units,
                                kernel_initializer='normal',
                                activation='linear'))
        adam = optimizers.Adam(lr=self.learning_rate)
        self.NN_model.compile(loss='mean_squared_error',
                              optimizer=adam,
                              metrics=['mean_squared_error'])

        # self.NN_model.summary()

        yaml_string = self.NN_model.to_yaml()
        # model = model_from_yaml(yaml_string)
        model_summary_dir = self.CONFIG['regression']['dirs']['tc']\
                ['neural_network']['model_summary']
        os.makedirs(model_summary_dir, exist_ok=True)
        model_summary_path = f'{model_summary_dir}model.yml'
        with open(model_summary_path, 'w') as outfile:
            yaml.dump(yaml_string, outfile, default_flow_style=False)

        checkpoint_root_dir = self.CONFIG['regression']['dirs']\
                ['tc']['neural_network']['checkpoint']

        self.checkpoint_dir = (f"""{checkpoint_root_dir}""")
                               # f"""{self.condition}/""")
        # Remove old checkpoint files
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        # Create a new checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        checkpoint_path = (f"""{self.checkpoint_dir}"""
                           f"""{checkpoint_name}""")

        checkpoint_callback = ModelCheckpoint(
            checkpoint_path, monitor='val_loss', verbose=0,
            save_best_only=True, mode='auto')

        log_dir = self.CONFIG['regression']['dirs']['tc']\
                ['neural_network']['logs']
        log_dir = f'{log_dir}{self.DNN_structure_str}/'
        # Remove old tensorboard files
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        tensorboard_callback = TrainValTensorBoard(
            log_dir=log_dir, histogram_freq=int(self.epochs / 10),
            batch_size=32, write_graph=True, write_grads=False,
            write_images=False,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None,
            update_freq='epoch')
        self.callbacks_list = [checkpoint_callback, tensorboard_callback]

    def read_smap_era5(self, sample, col_name, divide, large_or_small):
        self.logger.info((f"""Reading SMAP-ERA5 data"""))
        combined, target, test_target, train_length = \
                self.get_combined_data(sample, col_name, divide,
                                       large_or_small)

        if combined is None:
            return

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

    def get_combined_data(self, sample, col_name, divide, large_or_small):
        train, test = self._get_train_test(sample, col_name, divide,
                                           large_or_small)
        if train is None:
            return None, None, None, None,

        train_length = len(train)

        target = getattr(train, self.target_name)
        test_target = getattr(test, self.target_name)
        train.drop([self.target_name], axis=1, inplace=True)
        test.drop([self.target_name], axis=1, inplace=True)

        combined = train.append(test)
        combined.reset_index(inplace=True)
        combined.drop(['index'], axis=1, inplace=True)

        return combined, target, test_target, train_length

    def _get_train_test(self, sample, col_name, divide, large_or_small):
        years = [y for y in range(self.train_period[0].year,
                                  self.train_period[1].year+1)]
        # years = [2015, 2016, 2017, 2018, 2019]
        df = None
        for y in years:
            table_name = f'tc_smap_era5_{y}_{self.basin}'
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

        if sample:
            df = shuffle(df)
            rows, cols = df.shape
            cut_rows = int(rows * 0.3)
            df = df[:cut_rows]

        # if col_name is not None:
        #     df, self.condition = \
        #             utils.filter_dataframe_by_column_value_divide(
        #                 df, col_name, divide, large_or_small)
        # else:
        #     self.condition = 'all'

        # Fiiter outilers by z score
        z = np.abs(stats.zscore(df))
        threshold = 3
        outliers = np.where(z > threshold)

        cols = list(df.columns)
        cols_num = len(df.columns)
        group_size = 1
        groups_num = math.ceil(cols_num / group_size)
        fig_dir = self.CONFIG['result']['dirs']['fig']\
                ['hist_of_regression_features']['original']
        os.makedirs(fig_dir, exist_ok=True)
        for i in range(groups_num):
            try:
                start = i * group_size
                end = min(cols_num, (i + 1) * group_size)
                self.save_features_histogram(df, cols[start:end], fig_dir)
            except Exception as msg:
                breakpoint()
                exit(msg)

        if 'normalization' in self.instructions:
            # Normalization
            normalized_columns = df.columns.drop(self.target_name)
            scaler = MinMaxScaler()
            df[normalized_columns] = scaler.fit_transform(
                df[normalized_columns])

            fig_dir = self.CONFIG['result']['dirs']['fig']\
                    ['hist_of_regression_features']\
                    ['after_normalization']
            os.makedirs(fig_dir, exist_ok=True)
            for i in range(groups_num):
                try:
                    start = i * group_size
                    end = min(cols_num, (i + 1) * group_size)
                    self.save_features_histogram(df, cols[start:end],
                                                 fig_dir)
                except Exception as msg:
                    breakpoint()
                    exit(msg)

        train, test = train_test_split(df, test_size=0.2)

        if 'only_hist' in self.instructions:
            return None, None
        else:
            return train, test

    def save_features_histogram(self, df, columns, fig_dir):
        try:
            df.hist(column=columns, figsize = (12,10))
        except Exception as msg:
            breakpoint()
            exit(msg)

        fig_name = '-'.join(columns)
        fig_name = f'{fig_name}.png'
        plt.savefig(f'{fig_dir}{fig_name}')

    def _show_correlation_heatmap(self, data):
        C_mat = data.corr()
        fig = plt.figure(figsize=(15, 15))
        sb.heatmap(C_mat, vmax=.8, square=True)
        plt.show()
