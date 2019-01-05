"""General and miscellaneous utility functions."""

from collections import namedtuple

import numpy as np
from scipy.spatial.distance import cdist

from machine_learning.constants import RANDOM_SEED


DataTuple = namedtuple('DataTuple', ['X', 'y'])


def add_constant(data_tuple):
    """Adds a constant column to the independent variables.

    Arguments:
        data_tuple {DataTuple} -- DataTuple containing independent and dependent variables.

    Returns:
        DataTuple -- Data with added constant column.
    """

    X_with_constant = data_tuple.X.copy()
    X_with_constant['constant'] = np.ones(data_tuple.y.values.shape)
    new_data_tuple = DataTuple(X_with_constant, data_tuple.y)
    return new_data_tuple


def k_fold_split_indexes(N, k):
    """Returns indexes for a k-fold split.

    Arguments:
        N {int} -- Size of data.
        k {int} -- Number of folds.

    Returns:
        list -- List of randomly permutated indexes.
    """

    indexes = np.random.RandomState(RANDOM_SEED).permutation(N)
    splitted = np.array_split(indexes, k)
    return splitted


def get_k_nn(point, X, k):
    """Finds the k nearest neighbors of a point from a set of points.

    Arguments:
        point {ndarray} -- A single point.
        X {ndarray} -- A set of points.
        k {int} -- Number of neighbors to find.

    Returns:
        ndarray -- Indexes of the neighbors.
        ndarray -- Distances to the neighbors from point.
    """

    k = int(k)
    p_shape = point.shape
    X_shape = X.shape
    dists = cdist(X, point.reshape(1, p_shape[0]))
    neighbors = np.argpartition(dists.ravel(), k-1)[:k]
    dists = dists.reshape((X_shape[0], 1))
    return neighbors, dists[neighbors]
