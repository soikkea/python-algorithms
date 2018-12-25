import numpy as np

from machine_learning.utilities import add_constant


def regression(method, error_func, train, test):
    y_pred = method(train, test)
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


def MSE(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.reshape((-1, 1))
        y_true = y_true.reshape((-1, 1))
    N = y_true.size
    return np.sum((y_true - y_pred) ** 2) / N
