from collections import namedtuple

import numpy as np
from scipy.spatial.distance import cdist

from machine_learning.constants import RANDOM_SEED


DataTuple = namedtuple('DataTuple', ['X', 'y'])


def add_constant(data_tuple):
    X_with_constant = data_tuple.X.copy()
    X_with_constant['constant'] = np.ones(data_tuple.y.values.shape)
    new_data_tuple = DataTuple(X_with_constant, data_tuple.y)
    return new_data_tuple


def k_fold_split_indexes(N, k):
    indexes = np.random.RandomState(RANDOM_SEED).permutation(N)
    splitted = np.array_split(indexes, k)
    return splitted


def get_k_nn(point, X, k):
    k = int(k)
    p_shape = point.shape
    X_shape = X.shape
    dists = cdist(X, point.reshape(1, p_shape[0]))
    neighbors = np.argpartition(dists.ravel(), k-1)[:k]
    dists = dists.reshape((X_shape[0], 1))
    return neighbors, dists[neighbors]
