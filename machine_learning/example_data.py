"""Loads example data."""

import pandas as pd

from machine_learning.utilities import DataTuple


# Locations for the data
REG_TRAIN_DATA_LOC = "examples/data/regression_dataset_training.csv"
REG_TEST_DATA_LOC_1 = "examples/data/regression_dataset_testing.csv"
REG_TEST_DATA_LOC_2 = "examples/data/regression_dataset_testing_solution.csv"
CLASS_TRAIN_DATA_LOC = "examples/data/classification_dataset_training.csv"
CLASS_TEST_DATA_LOC_1 = "examples/data/classification_dataset_testing.csv"
CLASS_TEST_DATA_LOC_2 = "examples/data/classification_dataset_testing_solution.csv"

_train_data = pd.read_csv(REG_TRAIN_DATA_LOC, index_col=0)
_train_X = _train_data.iloc[:, :-1]
_train_y = _train_data.iloc[:, -1]
regression_train = DataTuple(_train_X, _train_y)

_test_X = pd.read_csv(REG_TEST_DATA_LOC_1, index_col=0)
_test_y = pd.read_csv(REG_TEST_DATA_LOC_2, index_col=0)
regression_test = DataTuple(_test_X, _test_y)

_train_data = pd.read_csv(CLASS_TRAIN_DATA_LOC, index_col=0)
_train_X = _train_data.iloc[:, :-1]
_train_y = _train_data.iloc[:, -1]
classification_train = DataTuple(_train_X, _train_y)

_test_X = pd.read_csv(CLASS_TEST_DATA_LOC_1, index_col=0)
_test_y = pd.read_csv(CLASS_TEST_DATA_LOC_2, index_col=0)
classification_test = DataTuple(_test_X, _test_y)
