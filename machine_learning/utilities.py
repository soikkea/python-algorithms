from collections import namedtuple

import numpy as np


DataTuple = namedtuple('DataTuple', ['X', 'y'])


def add_constant(data_tuple):
    X_with_constant = data_tuple.X.copy()
    X_with_constant['constant'] = np.ones(data_tuple.y.values.shape)
    new_data_tuple = DataTuple(X_with_constant, data_tuple.y)
    return new_data_tuple
