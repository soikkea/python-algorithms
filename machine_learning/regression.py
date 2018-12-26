import numpy as np

from machine_learning.constants import FOLDS, MAX_K
from machine_learning.utilities import (add_constant, get_k_nn, 
    k_fold_split_indexes)


def regression(method, error_func, train, test, **kwargs):
    y_pred = method(train, test, **kwargs)
    return error_func(y_pred, test.y.values)


def mean_regression(train, test):
    mean = mean_regression_fit(train.X, train.y)
    y_pred = mean_regression_predict(test.X, mean)
    return y_pred


def mean_regression_fit(X, y):
    return np.mean(y.values)


def mean_regression_predict(X, mean):
    rows = X.values.shape[0]
    y_pred = np.ones((rows, 1)) * mean
    return y_pred


def multivariate_regression(train, test):
    train = add_constant(train)
    test = add_constant(test)
    w_opt = multivariate_regression_fit(train.X, train.y)
    y_pred = multivariate_regression_predict(test.X, w_opt)
    return y_pred


def multivariate_regression_fit(X, y):
    X = X.values
    XTX = X.transpose().dot(X)
    y = y.values
    XTR = X.transpose().dot(y)
    w_opt = np.linalg.solve(XTX, XTR)
    return w_opt


def multivariate_regression_predict(X, w_opt):
    X = X.values
    y_pred = X.dot(w_opt)
    return y_pred


def backwards_feature_selection_regression(error_func, train, validate):
    train = add_constant(train)
    X_orig = train.X
    y = train.y
    validate = add_constant(validate)
    X_valid_orig = validate.X
    N_features = train.X.shape[1]
    features = list(range(N_features))
    # initialize
    min_error = np.infty
    best_set = None
    while len(features) > 1:
        X = X_orig.iloc[:, features]
        X_valid = X_valid_orig.iloc[:, features]
        w_opt = multivariate_regression_fit(X, y)
        y_pred = multivariate_regression_predict(X_valid, w_opt)
        error = error_func(y_pred, validate.y.values)
        if error < min_error:
            min_error = error
            best_set = features[:]
        assert len(w_opt[:-1]) < len(w_opt[:])
        features.pop(np.argmin(np.abs(w_opt[:-1])))        
    return min_error, best_set


def k_nn_regression(train, test, k):
    y_pred = k_nn_regression_predict(test.X, train.X, train.y, k)
    return y_pred


def k_nn_regression_fit(train, n_folds=FOLDS, max_k=MAX_K):
    X = train.X.values
    y = train.y.values
    N = X.shape[0]
    folds = k_fold_split_indexes(N, n_folds)
    min_error = np.infty
    best_k = 1
    for k in range(1, max_k):
        errors = np.zeros(n_folds)
        for i in range(n_folds):
            tmp_folds = folds[:]
            valid_ix = tmp_folds.pop(i)
            train_ix = np.concatenate(tmp_folds)
            y_train = y[train_ix]
            y_pred = k_nn_regression_predict(X[valid_ix, :], X[train_ix, :],
                                             y_train, k)
            mse = MSE(y_pred, y[valid_ix])
            errors[i] = (valid_ix.size * mse)
        mean_error = np.sum(errors) / N
        if mean_error < min_error:
            min_error = mean_error
            best_k = k
    return int(best_k), min_error


def k_nn_regression_predict(X, X_train, y_train, k):
    try:
        X = X.values
    except AttributeError:
        pass
    try:
        X_train = X.values
    except AttributeError:
        pass
    try:
        y_train = y_train.values
    except AttributeError:
        pass
    N = X.shape[0]
    y_train.reshape((-1, 1))
    y_pred = np.zeros((N, 1))
    for i in range(N):
        neighbors, dists = get_k_nn(X[i, :], X_train, k)
        # Add machine epsilon to avoid division by zero
        inv_dists = (1.0 / (dists + np.finfo(np.float).eps)).ravel()
        y_pred[i] = np.sum(inv_dists * y_train[neighbors]) / np.sum(inv_dists)
    return y_pred


def MSE(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.reshape((-1, 1))
        y_true = y_true.reshape((-1, 1))
    N = y_true.size
    return np.sum((y_true - y_pred) ** 2) / N
