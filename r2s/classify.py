import logging
import os
import pickle

import numpy as np
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
import lightgbm as lgb
from scipy.misc import derivative

import utils
from metrics import (focal_loss_lgb, focal_loss_lgb_eval_error,
                     lgb_f1_score, lgb_focal_f1_score, sigmoid)
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.utils import Bunch
from hyperopt import hp, tpe, fmin, Trials


Base = declarative_base()


class Classifier(object):

    def __init__(self, CONFIG, train_period, train_test_split_dt,
                 region, basin, passwd, save_disk, instructions,
                 smogn_target):
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

        utils.setup_database(self, Base)

        self.y_name = 'smap_windspd'
        self.model_dir = self.CONFIG["classify"]["dirs"]['lightgbm'][
            'model']

        self.set_variables()

        self.read_smap_era5()
        self.classify_threshold = 45
        self.label_y()
        self.lgb_train = lgb.Dataset(self.X_train, self.y_train,
                                     free_raw_data=False)
        self.lgb_eval = lgb.Dataset(self.X_test, self.y_test,
                                    reference=self.lgb_train,
                                    free_raw_data=False)

        if 'lgb' in self.instructions:
            self.record_best_params()
            if 'optimize' in self.instructions:
                self.optimize_classifier(maxevals=100)
            if 'compare' in self.instructions:
                self.compare_classifier()

            # elif 'compare_best' in self.instructions:
            #     self.compare_lgb_best_regressor()
            # elif 'best' in self.instructions:
            #     self.lgb_best_regressor()
            # elif 'default' in self.instructions:
            #     self.lgb_default_regressor()
            else:
                self.logger.error('Nothing happen')

    def record_best_params(self):
        result_dir = ('/Users/lujingze/Programming/SWFusion/'
                       'classify/tc/lightgbm/model/'
                       'na_valid_0.560000_45_fl_smogn_final'
                       '_unb_maxeval_2/')
        model_name = [f for f in os.listdir(result_dir)
                     if f.endswith('.pkl')
                     and f.startswith(f'{self.basin}')]
        if len(model_name) != 1:
            self.logger.error('Count of Bunch is not ONE')
            exit(1)

        with open(f'{result_dir}{model_name[0]}', 'rb') as f:
            regressor = pickle.load(f)

        best_params_df = pd.DataFrame.from_dict(
            regressor.best_params, orient='index',
            columns=['classifier'])
        best_params_df.to_csv(f'{result_dir}best_params.csv')

    def compare_classifier(self):
        classifier_root_path = (
            '/Users/lujingze/Programming/SWFusion/classify/'
            'tc/lightgbm/model/'
        )
        classifiers_dir_names = [
            'na_valid_0.688073_38_fl_smogn_final_unb_maxeval_2',
            'na_valid_0.703297_39_fl_smogn_final_unb_maxeval_2',
            'na_valid_0.630137_40_fl_smogn_final_unb_maxeval_2',
            'na_valid_0.555556_41_fl_smogn_final_unb_maxeval_2',
            'na_valid_0.555556_42_fl_smogn_final_unb_maxeval_2',
        ]
        candidates = {
            'diff_vote': [i for i in range(5)],
            'repeat_vote': [2] * 5,
            'single_vote': [2],
        }
        strategies = ['less_obey_more', 'sum', 'like_low', 'like_high']

        preds = utils.pred_of_classifier(self, classifier_root_path,
                                         classifiers_dir_names,
                                         candidates, strategies)
        acc = dict()
        f1 = dict()
        prec = dict()
        rec = dict()
        cm = dict()

        for idx, (key, val) in enumerate(preds.items()):
            acc[key] = accuracy_score(self.y_test, val)
            f1[key] = f1_score(self.y_test, val)
            prec[key] = precision_score(self.y_test, val)
            rec[key] = recall_score(self.y_test, val)
            cm[key] = confusion_matrix(self.y_test, val)

        schemes_num = len(preds.keys())
        metrics_num = 4
        rank = np.ndarray(shape=(schemes_num, metrics_num+1),
                          dtype=int)
        metrics_list = [acc, f1, prec, rec]
        schemes_name = list(preds.keys())

        for j, x in enumerate(metrics_list):
            sorted_x = {k: v for k, v in sorted(
                x.items(), key=lambda item: item[1],
                reverse=True)}
            val_descending = []
            rank_cursor = 0
            for i, (key, val) in enumerate(sorted_x.items()):
                if val not in val_descending:
                    val_descending.append(val)
                    rank_cursor += 1
                key_idx_in_schemes_name = schemes_name.index(key)
                rank[key_idx_in_schemes_name][j] = rank_cursor

        rank_sum_col_index = len(metrics_list)
        for i in range(len(schemes_name)):
            rank_sum = 0
            for j in range(rank_sum_col_index):
                rank_sum += rank[i][j]
            rank[i][rank_sum_col_index] = rank_sum

        rank_df = pd.DataFrame(data=rank, index=schemes_name,
                               columns=['acc', 'f1', 'prec',
                                        'rec', 'rank_sum'])
        breakpoint()

    def optimize_classifier(self, maxevals=200):
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
            focal_loss = lambda x,y: focal_loss_lgb(x, y, best['alpha'], best['gamma'])
            model = lgb.train(best, self.lgb_train, fobj=focal_loss)
            preds = model.predict(self.X_test)
            preds = sigmoid(preds)
            preds = (preds > 0.5).astype('int')
        else:
            model = lgb.train(best, self.lgb_train)
            preds = model.predict(self.X_test)
            preds = (preds > 0.5).astype('int')

        acc  = accuracy_score(self.y_test, preds)
        f1   = f1_score(self.y_test, preds)
        prec = precision_score(self.y_test, preds)
        rec  = recall_score(self.y_test, preds)
        cm   = confusion_matrix(self.y_test, preds)

        print((f"""acc: {acc:.4f}, f1 score: {f1:.4f}, """
               f"""precision: {prec:.4f}, recall: {rec:.4f}"""))
        print('confusion_matrix')
        print(cm)

        if self.save:
            results = Bunch(acc=acc, f1=f1, prec=prec, rec=rec, cm=cm)
            if ((hasattr(self, 'smogn_target')
                and self.smogn_target == 'train_splitted')
                    or (hasattr(self, 'valid'))):
                out_fname = f'{self.basin}_valid_{f1:.6f}'
            else:
                out_fname = f'{self.basin}_{f1:.6f}'

            out_fname += f'_{self.classify_threshold}'
            out_name_suffix = {
                'with_focal_loss': '_fl',
                'smogn': '_smogn',
                'smogn_final': '_smogn_final',
                'smogn_hyperopt': '_smogn_hyperopt',
                'is_unbalance': '_unb',
            }
            try:
                for idx, (key, val) in enumerate(
                        out_name_suffix.items()):
                    if hasattr(self, key) and getattr(self, key) is True:
                        out_fname += val
            except Exception as msg:
                breakpoint()
                exit(msg)
            out_dir = f'{self.model_dir}{out_fname}/'
            os.makedirs(out_dir, exist_ok=True)
            out_fname += '.pkl'
            results.model = model
            results.best_params = best
            pickle.dump(results, open(f'{out_dir}{out_fname}', 'wb'))

    def get_objective(self, train):

        def objective(params):
            """
            objective function for lightgbm.
            """
            # hyperopt casts as float
            params['num_boost_round'] = int(params['num_boost_round'])
            params['num_leaves'] = int(params['num_leaves'])

            # need to be passed as parameter
            if self.is_unbalance:
                params['is_unbalance'] = True
            params['verbose'] = -1
            params['seed'] = 1

            if self.with_focal_loss:
                focal_loss = lambda x, y: focal_loss_lgb(
                    x, y, params['alpha'], params['gamma'])

                cv_result = lgb.cv(
                    params,
                    train,
                    num_boost_round=params['num_boost_round'],
                    fobj=focal_loss,
                    feval=lgb_focal_f1_score,
                    stratified=True,
                    early_stopping_rounds=20)
            else:
                cv_result = lgb.cv(
                    params,
                    train,
                    num_boost_round=params['num_boost_round'],
                    metrics='binary_logloss',
                    feval=lgb_f1_score,
                    stratified=False,
                    early_stopping_rounds=20)

            self.early_stop_dict[objective.i] = len(
                cv_result['f1-mean'])
            score = round(cv_result['f1-mean'][-1], 4)

            objective.i += 1

            return -score

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
        if self.with_focal_loss:
            space['alpha'] = hp.uniform('alpha', 0.1, 0.75)
            space['gamma'] = hp.uniform('gamma', 0.5, 5)

        return space

    def read_smap_era5(self):
        self.logger.info((f"""Reading SMAP-ERA5 data"""))
        self.X_train, self.y_train, self.X_test, self.y_test = \
            utils.get_combined_data(self)

    def label_y(self):
        threshold = self.classify_threshold
        # interval = self.CONFIG['classify']['interval']
        try:
            y_train_class = [0] * len(self.y_train)
            for i in range(len(self.y_train)):
                val = self.y_train[i]
                y_train_class[i] = True if val > threshold else False
            self.y_train = pd.Series(data=y_train_class)

            y_test_class = [0] * len(self.y_test)
            for i in range(len(self.y_test)):
                val = self.y_test[i]
                y_test_class[i] = True if val > threshold else False
            self.y_test = pd.Series(data=y_test_class)
        except Exception as msg:
            breakpoint()
            exit(msg)

    def set_variables(self):
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

        if 'unbalance' in self.instructions:
            self.is_unbalance = True
        else:
            self.is_unbalance = False
