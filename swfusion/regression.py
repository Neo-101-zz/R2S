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
import seaborn as sns
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (VotingRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from joblib import dump, load

import utils
import era5
from metrics import (focal_loss_train, focal_loss_valid,
                     custom_asymmetric_train, custom_asymmetric_valid,
                     symmetric_train, symmetric_valid)

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
Base = declarative_base()


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

        self.edge = self.CONFIG['regression']['edge_in_degree']
        self.spa_resolu = self.CONFIG['era5']['spatial_resolution']
        self.lat_grid_points = [y * self.spa_resolu - 90 for y in range(
            self.CONFIG['era5']['lat_grid_points_number'])]
        self.lon_grid_points = [x * self.spa_resolu for x in range(
            self.CONFIG['era5']['lon_grid_points_number'])]

        self.model_dir = dict()
        self.model_dir['lgb'] = self.CONFIG['regression']['dirs'][
            'tc']['lightgbm']['model']

        self.validation_split = 0.2

        self.predict_table_name_prefix = 'predicted_smap_tc'

        self.compare_zorders = self.CONFIG['plot']['zorders'][
            'compare']

        self.wind_radii = self.CONFIG['wind_radii']

        utils.setup_database(self, Base)

        self.y_name = self.CONFIG['regression']['target']['smap_era5']
        # self.y_name = self.CONFIG['regression']['target']['sfmr_era5']

        self.set_variables()

        self.read_smap_era5()
        self.show_dataset_shape()

        self.lgb_train = lgb.Dataset(self.X_train, self.y_train,
                                     free_raw_data=False)
        self.lgb_eval = lgb.Dataset(self.X_test, self.y_test,
                                    reference=self.lgb_train,
                                    free_raw_data=False)

        if 'vote' in self.instructions:
            self.voting_regressor()
        if 'lgb' in self.instructions:
            if 'optimize' in self.instructions:
                self.optimize_lgb(maxevals=3)
            elif 'compare_best' in self.instructions:
                self.compare_lgb_best_regressor()
            elif 'best' in self.instructions:
                self.lgb_best_regressor()
            elif 'default' in self.instructions:
                self.lgb_default_regressor()
            else:
                self.logger.error('Nothing happen')

    def show_dataset_shape(self):
        print(f'X_train shape: {self.X_train.shape}')
        print(f'y_train shape: {self.y_train.shape}')
        print(f'X_test shape: {self.X_test.shape}')
        print(f'y_test shape: {self.y_test.shape}')

    def voting_regressor(self):
        estimators_num = 10
        regs = {
            'GBR': GradientBoostingRegressor(
                random_state=1, n_estimators=estimators_num),
            'RF': RandomForestRegressor(
                random_state=1, n_estimators=estimators_num,
                n_jobs=-1),
            'LR': LinearRegression(),
        }
        ereg_estimators = []
        ereg_name = ''
        for idx, (name, reg) in enumerate(regs.items()):
            ereg_estimators.append((name, reg))
            ereg_name += f'{name}_'

        ereg = VotingRegressor(estimators=ereg_estimators,
                               n_jobs=-1)
        ereg.fit(self.X_train, self.y_train)
        y_pred = ereg.predict(self.X_test)

        root_dir = ('/Users/lujingze/Programming/SWFusion/'
                    'regression/tc/lightgbm/model/')
        ereg_dir = f'{root_dir}{ereg_name[:-1]}/'
        os.makedirs(ereg_dir, exist_ok=True)

        dump(ereg, f'{ereg_dir}voting_model.joblib')

        with open(f'{ereg_dir}test_pred.pkl', 'wb') as f:
            pickle.dump(y_pred, f)

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
        utils.box_plot_windspd(default_dir, y_test, y_pred,
                               x_label='Wind speed range (m/s)',
                               y_label='Bias of wind speed (m/s)')
        mse = mean_squared_error(y_test, y_pred)
        print("MSE SCORE ON TEST DATA: {}".format(mse))
        print("RMSE SCORE ON TEST DATA: {}".format(math.sqrt(mse)))

        # save model to file
        pickle.dump(est, open((
            f"""{default_dir}"""
            f"""{self.basin}_mse_{mse}_{self.tag}"""
            f""".pickle.dat"""), 'wb'))

    def optimize_lgb(self, maxevals=200):
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
            focal_loss_obj = lambda x, y: focal_loss_train(
                x, y, self.y_max_pmf_reciprocal['y_train_splitted'],
                self.inversed_y_pmf_dict['y_train_splitted'], self.gamma)

            focal_loss_eval = lambda x, y: focal_loss_valid(
                x, y, self.y_max_pmf_reciprocal['y_train_splitted'],
                self.inversed_y_pmf_dict['y_train_splitted'], self.gamma)
            # focal_loss_obj = lambda x, y: focal_loss_train(x, y)
            # focal_loss_eval = lambda x, y: focal_loss_valid(x, y)

            model = lgb.train(best, self.lgb_train,
                              fobj=focal_loss_obj,
                              feval=focal_loss_eval)

            y_pred = model.predict(self.X_test)

            eval_name, eval_result, is_higher_better = \
                focal_loss_valid(y_pred, self.lgb_eval,
                                 self.y_max_pmf_reciprocal['y_test'],
                                 self.inversed_y_pmf_dict['y_test'],
                                 self.gamma)
            print((f"""---------------\n"""
                   f"""With focal loss\n"""
                   f"""---------------"""))
        else:
            model = lgb.train(best, self.lgb_train,
                              fobj=symmetric_train,
                              feval=symmetric_valid
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
            if ((hasattr(self, 'smogn_target')
                and self.smogn_target == 'train_splitted')
                    or (hasattr(self, 'valid'))):
                out_fname = f'{self.basin}_valid_{eval_result:.6f}'
            else:
                out_fname = f'{self.basin}_{eval_result:.6f}'
            out_name_suffix = {
                'with_focal_loss': '_fl',
                'smogn': '_smogn',
                'smogn_final': '_smogn_final',
                'smogn_hyperopt': '_smogn_hyperopt',
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

    def ensemble_lgb_regressor(self):
        try:
            root_dir = ('/Users/lujingze/Programming/SWFusion/'
                        'regression/tc/lightgbm/model/')
            model_dir = {
                'SG-FL': (f"""{root_dir}na_101.845662_fl_smogn_"""
                          f"""final_threshold_square_2/"""),
                'MSE': f'{root_dir}na_2.188733/',
            }
            er_name = ''
            estimators = []
            for idx, (name, out_dir) in enumerate(model_dir.items()):
                er_name += f'{name}_'
                save_file = [f for f in os.listdir(out_dir)
                             if f.endswith('.pkl')
                             and f.startswith(f'{self.basin}')]
                if len(save_file) != 1:
                    self.logger.error('Count of Bunch is not ONE')
                    exit(1)

                with open(f'{out_dir}{save_file[0]}', 'rb') as f:
                    best_result = pickle.load(f)

                estimators.append((name, best_result.model))

            er_name = er_name[:-1]
            er = VotingRegressor(estimators)
            er.fit(self.X_train, self.y_train)

            os.makedirs(f'{root_dir}{er_name[:-1]}/', exist_ok=True)
            y_pred = er.predict(self.X_test)
            y_pred.to_pickle(f'{er_dir}y_pred.pkl')
        except Exception as msg:
            breakpoint()
            exit(msg)


    def compare_lgb_best_regressor(self):
        regressor_paths = {
            'MSE': ('/Users/lujingze/Programming/SWFusion/regression/'
                    'tc/lightgbm/model/'
                    'na_valid_2.496193/'),
            'SMOGN-TCL': (
                '/Users/lujingze/Programming/SWFusion/'
                'regression/tc/lightgbm/model/'
                'na_valid_2557.909583_fl_smogn_final_thre_50_power_3_under_maxeval_100/'),
        }
        y_test = self.y_test.to_numpy()
        y_test_pred = dict()
        sum = np.zeros(shape=y_test.shape)
        name_comparison = ''
        for idx, (key, val) in enumerate(regressor_paths.items()):
            out_root_dir = val.split(f'{self.basin}_')[0]
            name = val.split('model/')[1].split('/')[0]
            name_comparison += f'_vs_{name}'

            model_name = [f for f in os.listdir(val)
                         if f.endswith('.pkl')
                         and f.startswith(f'{self.basin}')]
            if len(model_name) != 1:
                self.logger.error('Count of Bunch is not ONE')
                exit(1)

            with open(f'{val}{model_name[0]}', 'rb') as f:
                regressor = pickle.load(f)

            best_params_df = pd.DataFrame.from_dict(
                regressor.best_params, orient='index',
                columns=[key])
            best_params_df.to_csv(f'{val}best_params.csv')

            try:
                col_names_to_plot = self.X_train.columns.tolist()
                for idx, val in enumerate(col_names_to_plot):
                    col_names_to_plot[idx] = val.replace('_', ' ')

                feature_imp = pd.DataFrame(
                    sorted(
                        zip(regressor.model.feature_importance(
                            importance_type='split'),
                            col_names_to_plot)
                    ),
                    columns=['Value', 'Feature']
                )
                plt.figure(figsize=(20, 20))
                sns.barplot(x="Value", y="Feature",
                            data=feature_imp.sort_values(
                                by="Value", ascending=False))
                # plt.title('LightGBM Features (avg over folds)')
                plt.tight_layout()
                plt.savefig(f'{out_root_dir}lgbm_importances.png', dpi=600)
            except Exception as msg:
                breakpoint()
                exit(msg)

            y_test_pred[key] = regressor.model.predict(self.X_test)

            sum += y_test_pred[key]

        # y_test_pred['mean'] = sum / float(len(y_test_pred.keys()))
        # name_comparison += f'_vs_mean'

        if hasattr(self, 'classify') and self.classify:
            classifiers = []
            classifier_root_path = (
                '/Users/lujingze/Programming/SWFusion/classify/'
                'tc/lightgbm/model/'
            )
            classifiers_dir_names = [
                # 'na_valid_0.688073_38_fl_smogn_final_unb_maxeval_2',
                # 'na_valid_0.703297_39_fl_smogn_final_unb_maxeval_2',
                # 'na_valid_0.630137_40_fl_smogn_final_unb_maxeval_2',
                # 'na_valid_0.555556_41_fl_smogn_final_unb_maxeval_2',
                # 'na_valid_0.555556_42_fl_smogn_final_unb_maxeval_2',
                'na_valid_0.560000_45_fl_smogn_final_unb_maxeval_2',
            ]
            for name in classifiers_dir_names:
                classifiers_dir = f'{classifier_root_path}{name}/'
                save_file = [f for f in os.listdir(classifiers_dir)
                             if f.endswith('.pkl')
                             and f.startswith(f'{self.basin}')]
                if len(save_file) != 1:
                    self.logger.error('Count of Bunch is not ONE')
                    exit(1)

                with open(f'{classifiers_dir}{save_file[0]}', 'rb') as f:
                    best_result = pickle.load(f)

                classifiers.append(best_result.model)

            y_class_pred = np.zeros(shape=y_test.shape)

            classifiers_preds = []
            for clf in classifiers:
                classifiers_preds.append(clf.predict(self.X_test))
            for i in range(len(self.y_test)):
                high = True
                for pred in classifiers_preds:
                    if pred[i] < 0:
                        high = False
                        break
                if high:
                    y_class_pred[i] = 1
                else:
                    y_class_pred[i] = -1

            y_test_hybrid_pred = np.zeros(shape=y_test.shape)
            for i in range(len(y_test)):
                if y_class_pred[i] >= 0:
                    y_test_hybrid_pred[i] = y_test_pred[
                        'SMOGN-TCL'][i]
                else:
                    y_test_hybrid_pred[i] = y_test_pred[
                        'MSE'][i]

            y_test_pred['HYBRID'] = y_test_hybrid_pred
            name_comparison += f'_vs_hybrid'

        name_comparison = name_comparison[4:]
        if hasattr(self, 'valid') and self.valid:
            name_comparison = f'VALID_{name_comparison}'
        else:
            name_comparison = f'TEST_{name_comparison}'

        out_dir = f'{out_root_dir}{name_comparison}/'
        os.makedirs(out_dir, exist_ok=True)

        utils.box_plot_windspd(out_dir, y_test, y_test_pred,
                               x_label=('Actual SMAP wind speed '
                                        'range (m/s)'),
                               y_label='Bias of wind speed (m/s)')
        utils.scatter_plot_pred(out_dir, y_test, y_test_pred,
                                statistic=False,
                                x_label='Actual SMAP wind speed (m/s)',
                                y_label='Simulated SMAP wind speed (m/s)')
        utils.statistic_of_bias(y_test, y_test_pred)

    def lgb_best_regressor(self, dpi=600):
        out_dir = ('/Users/lujingze/Programming/SWFusion/regression/'
                   'tc/lightgbm/model/'
                   'na_valid_177.225788_fl_smogn_final/')
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

        try:
            col_names_to_plot = self.X_train.columns.tolist()
            for idx, val in enumerate(col_names_to_plot):
                col_names_to_plot[idx] = val.replace('_', ' ')

            feature_imp = pd.DataFrame(
                sorted(
                    zip(best_result.model.feature_importance(
                        importance_type='split'),
                        col_names_to_plot)
                ),
                columns=['Value', 'Feature']
            )
            plt.figure(figsize=(20, 20))
            sns.barplot(x="Value", y="Feature",
                        data=feature_imp.sort_values(
                            by="Value", ascending=False))
            # plt.title('LightGBM Features (avg over folds)')
            plt.tight_layout()
            plt.savefig(f'{out_dir}lgbm_importances.png', dpi=dpi)
        except Exception as msg:
            breakpoint()
            exit(msg)
        # fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        # lgb.plot_importance(best_result.model, ax, title=None)
        # plt.show()

        y_pred = best_result.model.predict(self.X_test)

        y_test = self.y_test.to_numpy()
        # utils.jointplot_kernel_dist_of_imbalance_windspd(
        #     out_dir, y_test, y_pred)
        # utils.box_plot_windspd(out_dir, y_test, y_pred)

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
                # focal_loss_obj = lambda x, y: focal_loss_train(x, y)
                # focal_loss_eval = lambda x, y: focal_loss_valid(x, y)
                focal_loss_obj = lambda x, y: focal_loss_train(
                    x, y, self.y_max_pmf_reciprocal['y_train_splitted'],
                    self.inversed_y_pmf_dict['y_train_splitted'], self.gamma)

                focal_loss_eval = lambda x, y: focal_loss_valid(
                    x, y, self.y_max_pmf_reciprocal['y_train_splitted'],
                    self.inversed_y_pmf_dict['y_train_splitted'], self.gamma)

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

    def read_smap_era5(self):
        self.logger.info((f"""Reading SMAP-ERA5 data"""))
        self.X_train, self.y_train, self.X_test, self.y_test = \
            utils.get_combined_data(self)

    def set_variables(self):
        if 'classify' in self.instructions:
            self.classify = True
        else:
            self.classify = False

        if 'valid' in self.instructions:
            self.valid = True
        else:
            self.valid = False

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
