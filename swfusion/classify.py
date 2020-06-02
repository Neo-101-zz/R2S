import logging

import numpy as np
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
import lightgbm as lgb
from scipy.misc import derivative

import utils
import bclassmetrics

Base = declarative_base()


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label

    def fl(x, t):
        p = 1/(1+np.exp(-x))
        return (
            -(a*t + (1-a)*(1-t))
            * ((1 - (t*p + (1-t)*(1-p)))**g)
            * (t*np.log(p)+(1-t)*np.log(1-p))
        )

    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)

    return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.label
    p = 1/(1+np.exp(-y_pred))
    loss = (
        -(a*y_true + (1-a)*(1-y_true))
        * ((1 - (y_true*p + (1-y_true)*(1-p)))**g)
        * (y_true*np.log(p) + (1-y_true)*np.log(1-p))
    )
    # (eval_name, eval_result, is_higher_better)
    return 'focal_loss', np.mean(loss), False


class Classifier(object):

    def __init__(self, CONFIG, train_period, train_test_split_dt,
                 region, basin, passwd, save_disk, instructions):
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

        utils.setup_database(self, Base)

        self.y_name = 'smap_windspd'

        if 'smogn' in self.instructions:
            self.smogn = True
        else:
            self.smogn = False

        if 'load' in self.instructions:
            self.load = True
        else:
            self.load = False

        if 'focus' in self.instructions:
            self.focus = True
        else:
            self.focus = False

        if 'borrow' in self.instructions:
            self.borrow = True
        else:
            self.borrow = False

        self.read_smap_era5(False)
        self.label_y()

        self.lgmb_classifier()

    def lgmb_classifier(self):
        # Default
        est = lgb.LGBMClassifier(is_unbalance=True)
        # Fit
        est.fit(self.X_train, self.y_train)
        # Predict
        y_test_pred = est.predict(self.X_test)
        # y_test_pred_proba = est.predict_proba(self.X_test)
        y_test_true = self.y_test.to_numpy()
        bclassmetrics.confusion_matrix_metrics(
            y_test_true, y_test_pred)

        # Set is_unbalanced=True
        est = lgb.LGBMClassifier(is_unbalance=True)
        # Fit
        est.fit(self.X_train, self.y_train)
        # Predict
        y_test_pred = est.predict(self.X_test)
        # y_test_pred_proba = est.predict_proba(self.X_test)
        y_test_true = self.y_test.to_numpy()
        bclassmetrics.confusion_matrix_metrics(
            y_test_true, y_test_pred)


        focal_loss = lambda x, y: focal_loss_lgb(x, y, alpha=0.25,
                                                gamma=1.)
        focal_loss_eval = lambda x, y: focal_loss_lgb_eval_error(
            x, y, alpha=0.25, gamma=1.)
        model = lgb.train(best, self.lgtrain, fobj=focal_loss, feval=focal_loss_eval)


        return

    def read_smap_era5(self, sample):
        self.logger.info((f"""Reading SMAP-ERA5 data"""))
        self.X_train, self.y_train, self.X_test, self.y_test = \
            utils.get_combined_data(self, sample)

    def split_combined(self, combined, train_length):
        train = combined[:train_length]
        test = combined[train_length:]

        return train, test

    def label_y(self):
        # interval = self.CONFIG['classify']['interval']
        threshold = 40

        try:
            y_train_class = [0] * len(self.y_train)
            for i in range(len(self.y_train)):
                val = self.y_train[i]
                y_train_class[i] = 1 if val > threshold else -1
            self.y_train = pd.Series(data=y_train_class)

            y_test_class = [0] * len(self.y_test)
            for i in range(len(self.y_test)):
                val = self.y_test[i]
                y_test_class[i] = 1 if val > threshold else -1
            self.y_test = pd.Series(data=y_test_class)
        except Exception as msg:
            breakpoint()
            exit(msg)
