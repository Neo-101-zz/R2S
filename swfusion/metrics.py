import time

import numpy as np
from sklearn.metrics import f1_score
from scipy.misc import derivative


def sigmoid(x): return 1./(1. + np.exp(-x))


def focal_loss_train(y_pred, train_data, reciprocal,
                     inversed_pmf_dict, gamma):
    try:
        y_true = train_data.label
        # st = time.time()
        # ratio = np.zeros(shape=y_pred.shape)
        # for i in range(len(y_pred)):
        #     ratio[i] = (reciprocal
        #                 * inversed_pmf_dict[int(y_true[i])]) ** gamma
        # used_time = time.time() - st
        # print(used_time)
        # breakpoint()
        ratio = (
            (np.vectorize(inversed_pmf_dict.get)(
                np.array(y_true, dtype=int))
                * reciprocal)) ** gamma

        residual = (y_true - y_pred).astype("float")
        grad = -2*ratio*residual
        hess = 2*ratio
    except Exception as msg:
        breakpoint()
        exit(msg)

    return grad, hess


def focal_loss_valid(y_pred, train_data, reciprocal,
                     inversed_pmf_dict, gamma):
    try:
        y_true = train_data.label
        # ratio = np.zeros(shape=y_pred.shape)
        # for i in range(len(y_pred)):
        #     ratio[i] = (reciprocal
        #                 * inversed_pmf_dict[int(y_true[i])]) ** gamma
        ratio = (
            (np.vectorize(inversed_pmf_dict.get)(
                np.array(y_true, dtype=int))
                * reciprocal)) ** gamma

        residual = (y_true - y_pred).astype("float")
        loss = (residual**2)*ratio
    except Exception as msg:
        breakpoint()
        exit(msg)

    return "custom_asymmetric_eval", np.mean(loss), False


"""
def focal_loss_train(y_pred, train_data, threshold=45, power=2.5):
    try:
        y_true = train_data.label
        residual = (y_true - y_pred).astype("float")
        # Only loss more when windspd larger than threshold
        alpha = np.where(y_true > threshold, y_true**power, 1)
        # When y_pred is low and windspd larger than threshold
        grad = np.where(residual > 0, -2*alpha*residual, -2*residual)
        hess = np.where(residual > 0, 2*alpha, 2.0)
        # grad = -2*alpha*residual
        # hess = 2*alpha
    except Exception as msg:
        breakpoint()
        exit(msg)

    return grad, hess


def focal_loss_valid(y_pred, train_data, threshold=45, power=2.5):
    try:
        y_true = train_data.label
        residual = (y_true - y_pred).astype("float")
        # Only loss more when windspd larger than threshold
        alpha = np.where(y_true > threshold, y_true**power, 1)
        # When y_pred is low and windspd larger than threshold
        loss = np.where(residual > 0, (residual**2)*alpha, residual**2)
        # loss = (residual**2)*alpha
    except Exception as msg:
        breakpoint()
        exit(msg)

    return "custom_asymmetric_eval", np.mean(loss), False
"""


def custom_asymmetric_train(y_pred, train_data, slope, min_alpha):
    try:
        y_true = train_data.label
        residual = (y_true - y_pred).astype("float")
        alpha = np.where(slope*y_true > min_alpha, slope*y_true,
                         min_alpha)
        grad = np.where(residual > 0, -2*alpha*residual, -2*residual)
        hess = np.where(residual > 0, 2*alpha, 2.0)
    except Exception as msg:
        breakpoint()
        exit(msg)

    return grad, hess


def custom_asymmetric_valid(y_pred, train_data, slope, min_alpha):
    try:
        y_true = train_data.label
        residual = (y_true - y_pred).astype("float")
        alpha = np.where(slope*y_true > min_alpha, slope*y_true,
                         min_alpha)
        loss = np.where(residual > 0, (residual**2)*alpha, residual**2)
    except Exception as msg:
        breakpoint()
        exit(msg)

    return "custom_asymmetric_eval", np.mean(loss), False


def symmetric_train(y_pred, train_data):
    y_true = train_data.label
    residual = (y_true - y_pred).astype("float")
    grad = np.full(y_pred.shape, -2*residual)
    hess = np.full(y_pred.shape, 2.0)

    return grad, hess


def symmetric_valid(y_pred, train_data):
    y_true = train_data.label
    residual = (y_true - y_pred).astype("float")
    loss = np.full(y_pred.shape, residual**2)

    return "symmetric_eval", np.mean(loss), False

def best_threshold(y_true, pred_proba, proba_range, verbose=False):
    """
    Function to find the probability threshold that optimises the f1_score

    Comment: this function is not used in this repo, but I include it in case the
    it useful

    Parameters:
    -----------
    y_true: numpy.ndarray
        array with the true labels
    pred_proba: numpy.ndarray
        array with the predicted probability
    proba_range: numpy.ndarray
        range of probabilities to explore.
        e.g. np.arange(0.1,0.9,0.01)

    Return:
    -----------
    tuple with the optimal threshold and the corresponding f1_score
    """
    scores = []
    for prob in proba_range:
        pred = [int(p>prob) for p in pred_proba]
        score = f1_score(y_true,pred)
        scores.append(score)
        if verbose:
            print("INFO: prob threshold: {}.  score :{}".format(round(prob,3), round(score,5)))
    best_score = scores[np.argmax(scores)]
    optimal_threshold = proba_range[np.argmax(scores)]
    return (optimal_threshold, best_score)


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p) + (1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    """
    Adapation of the Focal Loss for lightgbm to be used as evaluation loss

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    return 'focal_loss', np.mean(loss), False


def lgb_f1_score(preds, lgbDataset):
    """
    Implementation of the f1 score to be used as evaluation score for lightgbm

    Parameters:
    -----------
    preds: numpy.ndarray
        array with the predictions
    lgbDataset: lightgbm.Dataset
    """
    binary_preds = [int(p>0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return 'f1', f1_score(y_true, binary_preds), True


def lgb_focal_f1_score(preds, lgbDataset):
    """
    Adaptation of the implementation of the f1 score to be used as evaluation
    score for lightgbm. The adaptation is required since when using custom losses
    the row prediction needs to passed through a sigmoid to represent a
    probability

    Parameters:
    -----------
    preds: numpy.ndarray
        array with the predictions
    lgbDataset: lightgbm.Dataset
    """
    preds = sigmoid(preds)
    binary_preds = [int(p>0.5) for p in preds]
    y_true = lgbDataset.get_label()
    return 'f1', f1_score(y_true, binary_preds), True
