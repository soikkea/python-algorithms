"""Classification methods."""

import numpy as np

from machine_learning.constants import N_CLASSES, FOLDS, MAX_K, RANDOM_SEED
from machine_learning.utilities import k_fold_split_indexes, get_k_nn


def classification(method, error_func, train, test, **kwargs):
    """Perform classification for data and return error.

    Arguments:
        method {function} -- Classification method.
        error_func {function} -- Error function.
        train {DataTuple} -- Training data.
        test {DataTuple} -- Test data.

    All extra keyword arguments are passed to method.

    Returns:
        float -- Error value returned by error_func.
    """

    y_pred = method(train, test, **kwargs)
    return error_func(y_pred, test.y.values)


def max_classifier(train, test):
    """Maximum classifier.

    Classifies using the most common class in training data.

    Arguments:
        train {DataTuple} -- Training data.
        test {DataTuple} -- Test data.

    Returns:
        ndarray -- Predicted values.
    """

    max_category = max_classifier_fit(train.X, train.y)
    y_pred = max_classifier_predict(test.X, max_category)
    return y_pred


def max_classifier_fit(X, y):
    """Determines the most common class in input.

    Arguments:
        X {DataFrame} -- Indendent variables.
        y {DataFrame} -- Dependent variable.

    Returns:
        int -- Most common class.
    """

    y = y.values
    max_category = np.bincount(y.astype(int)).argmax()
    return max_category


def max_classifier_predict(X, max_category):
    """Classify using max classifier.

    Arguments:
        X {DataFrame} -- Independent variables.
        max_category {int} -- Class to classify to.

    Returns:
        ndarray -- Predicted values.
    """

    y_pred = np.ones((X.shape[0], 1), dtype=np.int) * max_category
    return y_pred


def multinomial_naive_bayes_classifier(train, test, n_classes=N_CLASSES):
    """Multinomial naive bayes classifier.

    See more at:
    https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes

    Arguments:
        train {DataTuple} -- Training data.
        test {DataTuple} -- Test data.

    Keyword Arguments:
        n_classes {int} -- Number of classes. (default: {N_CLASSES})

    Returns:
        ndarray -- Predicted values.
    """

    train_X = train.X.values
    train_y = train.y.values
    test_X = test.X.values
    class_priors, feature_likelihoods = mnb_classifier_fit(train_X, train_y,
                                                           n_classes)
    y_pred = mnb_classifier_predict(test_X, class_priors, feature_likelihoods)
    return y_pred


def mnb_classifier_fit(X, y, n_classes):
    """Fit MNB classifier.
    Calculates class priors and feature likelihoods.

    Arguments:
        X {ndarray} -- Independent variables.
        y {ndarray} -- Dependent variables.
        n_classes {int} -- Number of classes.

    Returns:
        ndarray -- Class priors.
        ndarray -- Feature likelihoods.
    """

    class_priors = mnb_class_priors(y, n_classes)
    feature_likelihoods = mnb_feature_likelihoods(X, y, n_classes)
    return class_priors, feature_likelihoods


def mnb_class_priors(y, n_classes):
    """Calculates the logaritm of the probability of belonging to each class.

    Arguments:
        y {ndarray} -- Class labels.
        n_classes {int} -- Number of class labels.

    Returns:
        ndarray -- Log of prior probabilities.
    """

    priors = np.zeros(n_classes)
    for c in range(n_classes):
        priors[c] = np.log(np.sum(y == c) / y.size)
    return priors


def mnb_feature_likelihoods(X, y, n_classes):
    """Calculates the probability of feature j, given class k, using Laplace smoothing.

    Arguments:
        X {ndarray} -- Features.
        y {ndarray} -- Class labels.
        n_classes {int} -- Number of classes.

    Returns:
        ndarray -- Logs of feature likelihoods.
    """

    n_features = X.shape[1]
    p_ij = np.zeros((n_classes, n_features))
    for c in range(n_classes):
        Fc_sum = np.sum(X[y == c, :])
        for j in range(n_features):
            Fnc = np.sum(X[y == c, j])
            p_ij[c, j] = np.log((1.0 + Fnc) / (n_features + Fc_sum))
    return p_ij


def mnb_classifier_predict(X, class_priors, feature_likelihoods):
    """Classify using MNB classifier.

    Arguments:
        X {ndarray} -- Independent variables.
        class_priors {ndarray} -- Class priors.
        feature_likelihoods {ndarray} -- Feature likelihoods.

    Returns:
        ndarray -- Predicted values.
    """

    n_classes = class_priors.size
    N = X.shape[0]
    posterior = np.zeros((N, n_classes))
    for i in range(N):
        posterior[i, :] = feature_likelihoods.dot(X[i, :])
    for c in range(n_classes):
        posterior[:, c] = posterior[:, c] + class_priors[c]
    y_pred = np.argmax(posterior, axis=1)
    return y_pred


def k_nn_classifier(train, test, k):
    """K-nearest neighbors classifier.

    Arguments:
        train {DataTuple} -- Training data.
        test {DataTuple} -- Test data.
        k {int} -- Value for k.

    Returns:
        ndarray -- Predicted values.
    """

    y_pred = k_nn_classifier_predict(test.X, train.X, train.y, k)
    return y_pred


def k_nn_classifier_fit(train, n_folds=FOLDS, max_k=MAX_K):
    """'Fit' K-nearest neighbors classifier by finding optimal value for k using cross validation.

    Arguments:
        train {DataTuple} -- Training data.

    Keyword Arguments:
        n_folds {int} -- Number of folds to use for validation. (default: {FOLDS})
        max_k {int} -- Maximum value for k. (default: {MAX_K})

    Returns:
        int -- Optimal value for k.
        float -- Error for selected k.
    """

    # TODO: combine with k_nn_regression_fit()?
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
            y_pred = k_nn_classifier_predict(X[valid_ix, :], X[train_ix, :],
                                             y[train_ix], k)
            error = classification_error(y_pred, y[valid_ix])
            errors[i] = (valid_ix.size * error)
        mean_error = np.sum(errors) / N
        if mean_error < min_error:
            min_error = mean_error
            best_k = k
    return int(best_k), min_error


def k_nn_classifier_predict(X, X_train, y_train, k, n_classes=N_CLASSES):
    """Classify using K-nearest neighbors classifier.

    Assigns class labels based on the most common class in k-nearest neighbors.

    Arguments:
        X {DataFrame} -- Independent variables.
        X_train {DataFrame} -- Independent training variables.
        y_train {DataFrame} -- Dependent training variables.
        k {int} -- Value of k.

    Keyword Arguments:
        n_classes {int} -- Number of classes. (default: {N_CLASSES})

    Returns:
        ndarray -- Predicted variables.
    """

    try:
        X = X.values
    except AttributeError:
        pass
    try:
        X_train = X_train.values
    except AttributeError:
        pass
    try:
        y_train = y_train.values
    except AttributeError:
        pass
    assert X.shape[1] == X_train.shape[1]
    N = X.shape[0]
    y_pred = np.zeros((N, 1))
    for i in range(N):
        point = X[i, :]
        neighbors, _ = get_k_nn(point, X_train, k)
        train_labels = y_train[neighbors]
        class_sums = [np.sum(train_labels == i) for i in range(n_classes)]
        y_pred[i] = k_nn_assign_label(class_sums)
    return y_pred


def k_nn_assign_label(class_sums):
    """Assing label according the most common class.
    If there are multiple candidates, pick one randomly.

    Arguments:
        class_sums {list} -- Class frequencies.

    Returns:
        int -- Assinged class label.
    """

    order = np.argsort(class_sums)[::-1]
    candidates = [x for x in order if x == order[0]]
    return np.random.RandomState(RANDOM_SEED).choice(candidates)


def classification_error(y_pred, y_true):
    """Return classification error.
    Sum of incorrectly assinged classes divided by the number of points.

    Arguments:
        y_pred {ndarray} -- Predicted values.
        y_true {ndarray} -- True values.

    Returns:
        float -- Error.
    """

    y_true = y_true.reshape(y_pred.shape)
    return np.sum(y_pred.astype(np.int)
                  != y_true.astype(np.int)) / float(y_pred.size)
