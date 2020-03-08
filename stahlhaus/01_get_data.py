#!/usr/bin/env python
import os
import tarfile
from six.moves import urllib
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np


DOWNLOAD_URL = "https://github.com/ageron/handson-ml/raw/master/"
HOUSING_PATH = "data"
HOUSING_URL = DOWNLOAD_URL + "datasets/housing/housing.tgz"

# download housing data, put into housing path,
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # create housing path if not exist
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    # build path to file
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # download and save
    urllib.request.urlretrieve(housing_url, tgz_path)
    # extract to housing path
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Pandas DataFrame docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# Pandas readcsv: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Download data and save
fetch_housing_data(HOUSING_URL, HOUSING_PATH)


# Load the data
data = load_housing_data()

data["income_category"] = np.ceil(data["median_income"] / 1.5)
data["income_category"].where(data["income_category"] < 5, 5.0, inplace=True)

# StratifiedShuffleSplit
#  Parameters
#     ----------
#     n_splits : int, default 10
#         Number of re-shuffling & splitting iterations.
#     test_size : float, int, None, optional (default=None)
#         If float, should be between 0.0 and 1.0 and represent the proportion
#         of the dataset to include in the test split. If int, represents the
#         absolute number of test samples. If None, the value is set to the
#         complement of the train size. If ``train_size`` is also None, it will
#         be set to 0.1.
#     train_size : float, int, or None, default is None
#         If float, should be between 0.0 and 1.0 and represent the
#         proportion of the dataset to include in the train split. If
#         int, represents the absolute number of train samples. If None,
#         the value is automatically set to the complement of the test size.
#     random_state : int, RandomState instance or None, optional (default=None)
#         If int, random_state is the seed used by the random number generator;
#         If RandomState instance, random_state is the random number generator;
#         If None, the random number generator is the RandomState instance used
#         by `np.random`.
spitting_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# list according to n_splits
for training_index, test_index in spitting_strategy.split(
    data, data["income_category"]
):
    training_data = data.loc[training_index]
    test_data = data.loc[test_index]

# Save unprepared test and training data
joblib.dump(training_data, "data/01_training_data_raw.pkl")
joblib.dump(test_data, "data/01_test_data_raw.pkl")
