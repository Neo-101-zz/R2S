import numpy as np


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
