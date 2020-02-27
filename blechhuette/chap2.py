#!/usr/bin/env python
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

# download: https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.tgz
DOWNLOAD_URL = "https://github.com/ageron/handson-ml/raw/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_URL + HOUSING_PATH + "/housing.tgz"


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


# expects data and a test ratio (0<ratio<1)
# returns random split train dataset and test dataset
# better implementation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
def split_testdata(data, test_ratio):
    # make this reproducible, always the same permutation
    np.random.seed(4711)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# Load the data
housing = load_housing_data()


# split using sklearn
# training_data, test_data = split_testdata(housing, 0.2), use sklearn
# training_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)

# stratify strategy: https://scikit-learn.org/stable/modules/cross_validation.html#stratified-shuffle-split

# median income has been crippled, categorize by dividing by 1.5, then make them discrete by joining all above 5 to category 5
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

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
for train_index, test_index in spitting_strategy.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Drop income category, no longer needed
for s in (strat_train_set, strat_test_set):
    s.drop("income_cat", axis=1, inplace=True)

# Override housing with training set
housing = strat_train_set.copy()

# A simple scatter plot of districts
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# A fancier versio with district population marker-size-coded and
# with house value color-coded
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"] / 100,
    label="population",
    figsize=(20, 12),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.legend()

# Compute a correlation matrix
corr_matrix = housing.corr()

# Plot a scatter matrix plot
attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]
scatter_matrix(housing[attributes], figsize=(20, 12))

# Add new variables (combinations of variables)
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_rooms"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# Compute a correlation matrix
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort()

# Override housing again, this time using training set without housing value variable
housing = strat_train_set.drop("median_house_value", axis=1)

# Save housing values as labels
housing_labels = strat_train_set["median_house_value"].copy()

# Data cleaning: Remove nan values (There are missing values in the total_bedrooms variable)
housing.dropna(
    subset=["total_bedrooms"]
)  # Remove dataframe entries where total_bedrooms is na
housing.drop("total_bedrooms", axis=1)  # Remove entire total_bedrooms column
median = housing["total_bedrooms"].median()  # Calculate median
housing["total_bedrooms"].fillna(median, inplace=True)  # Replace na with median value

# As an alternative, the Scikit-Learn class Imputer can take care of missing values
from sklearn.impute import SimpleImputer

# Init
imputer = SimpleImputer(missing_values=np.nan, strategy="median")

# Get dataframe with na again
housing = strat_train_set.drop("median_house_value", axis=1)

# Remove categorial variable, as median na replacement works only on numerical data
housing_num = housing.drop("ocean_proximity", axis=1)

# Fit imputer
imputer.fit(housing_num)

# Transform training set by replacing na (result X is numpy array)
X = imputer.transform(housing_num)

# Typecast back to DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# Convert the categorial variable ocean_proximity to numerical values
housing_cat = housing["ocean_proximity"]
housing_cat_encoded, housing_categories = housing_cat.factorize()

# These numerical values, however, may be misinterpreted in terms of similarity/ adjacency
# Solution is one-hot encoding, i.e. a binary variable (0-1) for each category
