import numpy as np


def regression(method, error_func, train, test):
    y_pred = method(train, test)
    return error_func(y_pred, test.y.values)


def mean_regression(train, test):
    mean = mean_regression_fit(train)
    y_pred = mean_regression_predict(test, mean)
    return y_pred


def mean_regression_fit(train):
    return np.mean(train.y.values)


def mean_regression_predict(test, mean):
    y = test.y.values
    y_pred = np.ones(y.shape) * mean
    return y_pred


def multivariate_regression(train, test):
    w_opt = multivariate_regression_fit(train)
    y_pred = multivariate_regression_predict(test, w_opt)
    return y_pred


def multivariate_regression_fit(train):
    X = train.X.values
    XTX = X.transpose().dot(X)
    y = train.y.values
    XTR = X.transpose().dot(y)
    w_opt = np.linalg.solve(XTX, XTR)
    return w_opt


def multivariate_regression_predict(test, w_opt):
    X = test.X.values
    y_pred = X.dot(w_opt)
    return y_pred


def MSE(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        y_pred = y_pred.reshape((-1, 1))
        y_true = y_true.reshape((-1, 1))
    N = y_true.size
    return np.sum((y_true - y_pred) ** 2) / N
