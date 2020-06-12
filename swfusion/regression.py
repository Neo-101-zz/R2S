import logging
import math
import os
import pickle
import shutil
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy.ext.declarative import declarative_base
import matplotlib.pyplot as plt
import seaborn as sb
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.layers import Dense
import xgboost as xgb
import lightgbm as lgb
import yaml
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval
from sklearn.utils import Bunch

import utils
import era5
from metrics import (custom_asymmetric_train, custom_asymmetric_valid,
                     symmetric_train, symmetric_valid)

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
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

    def __init__(self, CONFIG, train_period, train_test_split_dt,
                 region, basin, passwd, save_disk, instructions,
                 smogn_target, tag):
        self.logger = logging.getLogger(__name__)
        self.CONFIG = CONFIG
        self.train_period = train_period
        self.train_test_split_dt = train_test_split_dt
        self.region = region
        self.db_root_passwd = passwd
        self.engine = None
        self.session = None
        self.save_disk = save_disk
        self.basin = basin
        self.instructions = instructions
        self.smogn_target = smogn_target
        self.tag = tag

        self.smap_columns = ['x', 'y', 'lon', 'lat', 'windspd']
        # self.era5_columns = self.CONFIG['era5']['all_vars']

        self.era5_useless_columns = [
            'key', 'x_y'
        ]
        self.edge = self.CONFIG['regression']['edge_in_degree']
        self.spa_resolu = self.CONFIG['era5']['spatial_resolution']
        self.lat_grid_points = [y * self.spa_resolu - 90 for y in range(
            self.CONFIG['era5']['lat_grid_points_number'])]
        self.lon_grid_points = [x * self.spa_resolu for x in range(
            self.CONFIG['era5']['lon_grid_points_number'])]

        self.model_dir = dict()
        self.model_dir['xgb'] = self.CONFIG['regression']['dirs'][
            'tc']['xgboost']['model']
        self.model_dir['lgb'] = self.CONFIG['regression']['dirs'][
            'tc']['lightgbm']['model']

        self.validation_split = 0.2

        self.predict_table_name_prefix = 'predicted_smap_tc'

        self.compare_zorders = self.CONFIG['plot']['zorders'][
            'compare']

        self.wind_radii = self.CONFIG['wind_radii']

        utils.setup_database(self, Base)

        self.y_name = self.CONFIG['regression']['target']['smap_era5']

        self.set_variables()

        self.read_smap_era5()
        self.show_dataset_shape()

        # with open(('/Users/lujingze/Programming/SWFusion/regression/'
        #            'tc/dataset/comparison_smogn/test_smogn.pkl'),
        #           'rb') as f:
        #     test = pickle.load(f)
        # self.y_test = getattr(test, self.y_name).reset_index(drop=True)
        # self.X_test = test.drop([self.y_name], axis=1).reset_index()
        # self.show_dataset_shape()

        if 'dt' in self.instructions:
            self.decision_tree()
        if 'xgb' in self.instructions:
            # self.evaluation_df_visualition()
            self.xgb_regressor()
            # self.xgb_importance()
        if 'lgb' in self.instructions:
            if 'optimize' in self.instructions:
                self.optimize_lgb(maxevals=60)
            elif 'compare_best' in self.instructions:
                self.compare_lgb_best_regressor()
            elif 'best' in self.instructions:
                self.lgb_best_regressor()
            elif 'default' in self.instructions:
                self.lgb_default_regressor()
            else:
                self.logger.error('Nothing happen')
        if 'dnn' in self.instructions:
            self.deep_neural_network()

    def show_dataset_shape(self):
        print(f'X_train shape: {self.X_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print(f'X_test shape: {self.X_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')

    def set_variables(self):
        if 'plot_dist' in self.instructions:
            self.plot_dist = True
        else:
            self.plot_dist = False

        if 'smogn_hyperopt' in self.instructions:
            self.smogn_hyperopt = True
        else:
            self.smogn_hyperopt = False

        if 'smogn_final' in self.instructions:
            self.smogn_final = True
        else:
            self.smogn_final = False

        if 'save' in self.instructions:
            self.save = True
        else:
            self.save = False

        if 'load' in self.instructions:
            self.load = True
        else:
            self.load = False

        if 'focus' in self.instructions:
            self.with_focal_loss = True
        else:
            self.with_focal_loss = False

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
        fig_dir = self.CONFIG['regression']['dirs']['tc'][
            'xgboost']['importance']
        os.makedirs(fig_dir, exist_ok=True)
        selected_features_num = int(self.X_train.shape[1] / 3)

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
        model.fit(self.X_train, self.y_train)

        self.logger.info((f"""Evaluating on test set"""))
        predicted = model.predict(self.X_test)
        truth = self.y_test.to_numpy()
        mse = mean_squared_error(truth, predicted)
        print(f'MSE: {mse}')
        # one_test['mse'] = mse

        # save model to file
        pickle.dump(model, open((
            f"""{self.model_dir["xgb"]}"""
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
    def evaluate(self, est, params, X, y):
        # Choices
        if 'choices' in params.keys():
            params['learning_rate'] = params['choices'][
                'learning_rate']
            params['n_estimators'] = params['choices']['n_estimators']
            params.pop('choices')

        # Set params
        est.set_params(**params)

        # Calc CV score
        scores = cross_val_score(estimator=est, X=X, y=y,
                                 scoring='neg_mean_squared_error',
                                 cv=4)
        score = np.mean(scores)

        return score

    def lgb_default_regressor(self):
        self.logger.info((f"""Default lightgbm regressor"""))
        # Initilize instance of estimator
        est = lgb.LGBMRegressor()

        # Fit
        est.fit(self.X_train, self.y_train)
        # Predict
        y_pred = est.predict(self.X_test)

        y_test = self.y_test.to_numpy()

        if self.smogn:
            if self.smogn_hyperopt:
                default_dir = self.smogn_setting_dir
            else:
                default_dir = (f"""{self.model_dir["lgb"]}"""
                               f"""default_regressor/""")
        else:
            self.error(('Have not considered the situation that '
                        'not using SMOGN'))
            exit(1)

        os.makedirs(default_dir, exist_ok=True)
        # utils.distplot_imbalance_windspd(y_test, y_pred)
        utils.jointplot_kernel_dist_of_imbalance_windspd(
            default_dir, y_test, y_pred)
        utils.box_plot_windspd(default_dir, y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE SCORE ON TEST DATA: {}".format(mse))
        print("RMSE SCORE ON TEST DATA: {}".format(math.sqrt(mse)))

        # save model to file
        pickle.dump(est, open((
            f"""{default_dir}"""
            f"""{self.basin}_mse_{mse}_{self.tag}"""
            f""".pickle.dat"""), 'wb'))

    def optimize_lgb(self, maxevals=200):
        self.lgb_train = lgb.Dataset(self.X_train, self.y_train,
                                     free_raw_data=False)
        self.lgb_eval = lgb.Dataset(self.X_test, self.y_test,
                                    reference=self.lgb_train,
                                    free_raw_data=False)
        self.early_stop_dict = {}

        param_space = self.hyperparameter_space()
        objective = self.get_objective(self.lgb_train)
        objective.i = 0
        trials = Trials()
        best = fmin(fn=objective,
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=maxevals,
                    trials=trials)

        best['num_boost_round'] = self.early_stop_dict[
            trials.best_trial['tid']]
        best['num_leaves'] = int(best['num_leaves'])
        best['verbose'] = -1

        if self.with_focal_loss:
            focal_loss_obj = lambda x, y: custom_asymmetric_train(x, y)
            # focal_loss_obj = lambda x, y: custom_asymmetric_train(
            #     x, y, best['slope'], best['min_alpha'])
            # focal_loss_eval = lambda x, y: custom_asymmetric_valid(
            #     x, y, best['slope'], best['min_alpha'])

            model = lgb.train(best, self.lgb_train,
                              fobj=focal_loss_obj,
                              # feval=focal_loss_eval
                             )
            y_pred = model.predict(self.X_test)
            # eval_name, eval_result, is_higher_better = \
            #     custom_asymmetric_valid(y_pred, self.lgb_eval,
            #                             best['slope'],
            #                             best['min_alpha'])
            eval_name, eval_result, is_higher_better = \
                custom_asymmetric_valid(y_pred, self.lgb_eval)
            print((f"""---------------\n"""
                   f"""With focal loss\n"""
                   f"""---------------"""))
        else:
            model = lgb.train(best, self.lgb_train,
                              fobj=symmetric_train,
                              # feval=symmetric_valid
                             )
            y_pred = model.predict(self.X_test)
            eval_name, eval_result, is_higher_better = \
                symmetric_valid(y_pred, self.lgb_eval)
            print((f"""------------------\n"""
                   f"""Without focal loss\n"""
                   f"""------------------"""))

        print((f"""{eval_name}-mean: {eval_result}\n"""
               f"""is_higher_better: {is_higher_better}"""))

        if self.save:
            results = Bunch()
            out_fname = f'{self.basin}_{eval_result:.6f}'
            out_name_suffix = {
                'with_focal_loss': '_fl',
                'center': '_center',
                'smogn': '_smogn',
                'smogn_final': '_smogn_final',
                'smogn_hyperopt': '_smogn_hyperopt',
                'borrow': '_borrow',
            }
            try:
                for idx, (key, val) in enumerate(
                        out_name_suffix.items()):
                    if hasattr(self, key) and getattr(self, key) is True:
                        out_fname += val
            except Exception as msg:
                breakpoint()
                exit(msg)
            out_dir = f'{self.model_dir["lgb"]}{out_fname}/'
            os.makedirs(out_dir, exist_ok=True)
            out_fname += '.pkl'
            results.model = model
            results.best_params = best
            pickle.dump(results, open(f'{out_dir}{out_fname}', 'wb'))

        self.best = best
        self.model = model

    def compare_lgb_best_regressor(self):
        y_test_pred_paths = {
            'FL': '/Users/lujingze/Programming/SWFusion/regression/tc/lightgbm/model/na_3.050352_fl_smogn_final/test_pred.pkl',
            'MSE': '/Users/lujingze/Programming/SWFusion/regression/tc/lightgbm/model/na_2.114560/test_pred.pkl'
        }
        y_test_pred = dict()
        name_comparison = ''
        for idx, (key, val) in enumerate(y_test_pred_paths.items()):
            out_root_dir = val.split(f'{self.basin}_')[0]
            name = val.split('model/')[1].split('/test')[0]
            name_comparison += f'_vs_{name}'
            with open(val, 'rb') as f:
                y_test_pred[key] = pickle.load(f)
        name_comparison = name_comparison[4:]
        out_dir = f'{out_root_dir}{name_comparison}/'
        os.makedirs(out_dir, exist_ok=True)

        y_test = self.y_test.to_numpy()
        utils.box_plot_windspd(out_dir, y_test, y_test_pred)
        utils.statistic_of_bias(y_test, y_test_pred)

    def lgb_best_regressor(self):
        out_dir = ('/Users/lujingze/Programming/SWFusion/regression/'
                   'tc/lightgbm/model/'
                   'na_107.015567_fl_smogn_final/')
        save_file = [f for f in os.listdir(out_dir)
                     if f.endswith('.pkl')
                     and f.startswith(f'{self.basin}')]
        if len(save_file) != 1:
            self.logger.error('Count of Bunch is not ONE')
            exit(1)

        with open(f'{out_dir}{save_file[0]}', 'rb') as f:
            best_result = pickle.load(f)

        print((f"""-----------\n"""
               f"""Best result\n"""
               f"""-----------"""))
        print(best_result)

        y_pred = best_result.model.predict(self.X_test)
        y_test = self.y_test.to_numpy()
        utils.jointplot_kernel_dist_of_imbalance_windspd(
            out_dir, y_test, y_pred)
        utils.box_plot_windspd(out_dir, y_test, y_pred)

        with open(f'{out_dir}test_pred.pkl', 'wb') as f:
            pickle.dump(y_pred, f)

    def get_objective(self, train):

        def objective(params):
            """
            objective function for lightgbm.
            """
            # hyperopt casts as float
            params['num_boost_round'] = int(params['num_boost_round'])
            params['num_leaves'] = int(params['num_leaves'])

            # need to be passed as parameter
            params['verbose'] = -1
            params['seed'] = 1

            if self.with_focal_loss:
                focal_loss_obj = lambda x, y: custom_asymmetric_train(x, y)
                focal_loss_eval = lambda x, y: custom_asymmetric_valid(x, y)
                # focal_loss_obj = lambda x, y: custom_asymmetric_train(
                #     x, y, params['slope'], params['min_alpha'])
                # focal_loss_eval = lambda x, y: custom_asymmetric_valid(
                #     x, y, params['slope'], params['min_alpha'])

                cv_result = lgb.cv(
                    params,
                    train,
                    num_boost_round=params['num_boost_round'],
                    fobj=focal_loss_obj,
                    feval=focal_loss_eval,
                    stratified=False,
                    early_stopping_rounds=20)

                self.early_stop_dict[objective.i] = len(
                    cv_result['custom_asymmetric_eval-mean'])
                score = round(cv_result['custom_asymmetric_eval-mean'][
                    -1], 4)
            else:
                cv_result = lgb.cv(
                    params,
                    train,
                    num_boost_round=params['num_boost_round'],
                    fobj=symmetric_train,
                    feval=symmetric_valid,
                    stratified=False,
                    early_stopping_rounds=20)

                self.early_stop_dict[objective.i] = len(
                    cv_result['symmetric_eval-mean'])
                score = round(cv_result['symmetric_eval-mean'][-1], 4)

            objective.i += 1

            return score

        return objective

    def hyperparameter_space(self):
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'num_boost_round': hp.quniform('num_boost_round',
                                           50, 500, 20),
            'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
            'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
            'subsample': hp.uniform('subsample', 0.5, 1.),
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
            'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
        }
        # if self.with_focal_loss:
        #     space['slope'] = hp.uniform('slope', 0.05, 1.)
        #     space['min_alpha'] = hp.uniform('min_alpha', 1., 2.)

        return space

    def lgb_hyperoptimization_old(self):
        # NOTE: reference: http://dkopczyk.quantee.co.uk/hyperparameter-optimization/
        self.logger.info((f"""Hyperoptimize lightgbm regressor"""))

        # Define searched space
        # hp.uniform('subsample', 0.6, 0.4) ?
        # num_leaves < 2^(max_depth), ratio may be 0.618 ?
        hyper_space_tree = {
            'choices': hp.choice('choices', [
                {'learning_rate': 0.01,
                 'n_estimators': hp.randint(
                     'n_estimators_small_lr', 2000)},
                {'learning_rate': 0.1,
                 'n_estimators': 1000 + hp.randint(
                     'n_estimators_medium_lr', 2000)}
                ]),
            'max_depth':  hp.choice('max_depth', [7, 8, 9, -1]),
            'num_leaves': hp.choice('num_leaves', [127, 255, 511]),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }

        # Initilize instance of estimator
        est = lgb.LGBMRegressor(boosting_type='gbdt', n_jobs=-1,
                                 random_state=2018)

        # Objective minizmied
        hyperopt_objective = lambda params: (-1.0) * self.evaluate(
            est, params, self.X_train, self.y_train)

        # Trail
        trials = Trials()

        # Set algoritm parameters
        # TODO: try to just use tpe.suggest
        algo = partial(tpe.suggest,
                       n_startup_jobs=20, gamma=0.25,
                       n_EI_candidates=24)

        # Fit Tree Parzen Estimator
        best_vals = fmin(hyperopt_objective, space=hyper_space_tree,
                         algo=algo, max_evals=30, trials=trials,
                         rstate=np.random.RandomState(seed=2020))

        # Print best parameters
        best_params = space_eval(hyper_space_tree, best_vals)
        print("BEST PARAMETERS: " + str(best_params))

        # Print best CV score
        scores = [trial['result']['loss'] for trial in trials.trials]
        best_score = np.min(scores)
        print(f'BEST CV MSE: {best_score}')
        print(f'BEST CV RMSE: {math.sqrt(best_score)}')

        # Print execution time
        tdiff = (trials.trials[-1]['book_time']
                 - trials.trials[0]['book_time'])
        print(f'ELAPSED TIME (MINUTE): {tdiff.total_seconds() / 60}')

        choices = best_params['choices']
        best_params['learning_rate'] = choices['learning_rate']
        best_params['n_estimators'] = choices['n_estimators']
        del best_params['choices']

        # Set params
        est.set_params(**best_params)

        # Fit
        est.fit(self.X_train, self.y_train)
        y_pred = est.predict(self.X_test)

        # Predict
        y_test = self.y_test.to_numpy()
        mse = mean_squared_error(y_test, y_pred)
        print("MSE SCORE ON TEST DATA: {}".format(mse))
        print("RMSE SCORE ON TEST DATA: {}".format(math.sqrt(mse)))

    def decision_tree(self):
        self.logger.info((f"""Regressing with decision tree"""))
        self.logger.info((f"""Evaluating on test set"""))

        for depth in range(3, 20):
            tree = DecisionTreeRegressor(criterion='mse',
                                         max_depth=depth)
            tree.fit(self.X_train, self.y_train)

            predicted = tree.predict(self.X_test)
            truth = self.y_test.to_numpy()
            mse = mean_squared_error(truth, predicted)

            print(f"""Max depth {depth}: MSE {mse:.2f}""")

    def nn_hyperoptimization(self):
        self.logger.info((f"""Hyperoptimize neural network"""))

        hyper_space_tree = {
            'choices': hp.choice('choices', [
                {'learning_rate': 0.01,
                 'n_estimators': hp.randint(
                     'n_estimators_small_lr', 2000)},
                {'learning_rate': 0.1,
                 'n_estimators': 1000 + hp.randint(
                     'n_estimators_medium_lr', 2000)}
                ]),
            'max_depth':  hp.choice('max_depth', [7, 8, 9, -1]),
            'num_leaves': hp.choice('num_leaves', [127, 255, 511]),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }

    def deep_neural_network(self):
        """
        At least 4 variables to trail and error:

        Number of hidden layers
        -----------------------
        3, 6, 9

        Number of neurons
        -----------------
        Plan A: 1/3, 2/3, 3/3 of the size of input layer, plus the
        size of output layer.
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
        #     int(self.X_train.shape[1] * x / 3) for x in range(1, 4)
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
                score = self.NN_model.evaluate(self.X_test,
                                               self.y_test,
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
        era5_ = era5.ERA5Manager(self.CONFIG, self.period_test,
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
        # Traverse all ibtracs record within period_test
        tc_table_name = self.CONFIG['ibtracs']['table_name']
        TCTable = utils.get_class_by_tablename(self.engine,
                                               tc_table_name)
        for row in self.session.query(TCTable).filter(
            TCTable.date_time >= self.period_test[0],
            TCTable.date_time <= self.period_test[1]).yield_per(
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
                self.X_train, self.y_train, epochs=self.epochs,
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
                                input_dim=self.X_train.shape[1],
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
        model_summary_dir = self.CONFIG['regression']['dirs']['tc'][
            'neural_network']['model_summary']
        os.makedirs(model_summary_dir, exist_ok=True)
        model_summary_path = f'{model_summary_dir}model.yml'
        with open(model_summary_path, 'w') as outfile:
            yaml.dump(yaml_string, outfile, default_flow_style=False)

        checkpoint_root_dir = self.CONFIG['regression']['dirs'][
            'tc']['neural_network']['checkpoint']

        self.checkpoint_dir = (f"""{checkpoint_root_dir}""")
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

        log_dir = self.CONFIG['regression']['dirs']['tc'][
            'neural_network']['logs']
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
        self.callbacks_list = [checkpoint_callback,
                               tensorboard_callback]

    def read_smap_era5(self):
        self.logger.info((f"""Reading SMAP-ERA5 data"""))
        self.X_train, self.y_train, self.X_test, self.y_test = \
            utils.get_combined_data(self)
