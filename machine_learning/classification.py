import numpy as np


def classification(method, error_func, train, test, **kwargs):
    y_pred = method(train, test, **kwargs)
    return error_func(y_pred, test.y.values)


def max_classifier(train, test):
    max_category = max_classifier_fit(train.X, train.y)
    y_pred = max_classifier_predict(test.X, max_category)
    return y_pred


def max_classifier_fit(X, y):
    y = y.values
    max_category = np.bincount(y.astype(int)).argmax()
    return max_category


def max_classifier_predict(X, max_category):
    y_pred = np.ones((X.shape[0], 1), dtype=np.int) * max_category
    return y_pred


def classification_error(y_pred, y_true):
    y_true = y_true.reshape(y_pred.shape)
    return np.sum(y_pred.astype(np.int)
                  != y_true.astype(np.int)) / float(y_pred.size)
