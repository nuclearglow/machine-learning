#!/usr/bin/env python
import os
import tarfile
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import util
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from transformers.CombinedAttributesAdder import CombinedAttributesAdder
from transformers.DataFrameSelector import DataFrameSelector

HOUSING_PATH = "datasets/housing"

# Pandas DataFrame docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# Pandas readcsv: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Load the data
housing = load_housing_data()

# median income has been crippled, categorize by dividing by 1.5, then make them discrete by joining all above 5 to category 5
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

spitting_strategy = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# list according to n_splits
for train_index, test_index in spitting_strategy.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Drop income category, no longer needed
for s in (strat_train_set, strat_test_set):
    s.drop("income_cat", axis=1, inplace=True)

# Data cleanup: remove data which used to train/label against later
housing = strat_train_set.drop("median_house_value", axis=1)

category_attributes = ["ocean_proximity"]
numeric_attributes = util.difference(list(housing), category_attributes)

# Define our pipeline
numeric_pipeline = Pipeline(
    [
        ("selector", DataFrameSelector(numeric_attributes)),
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

category_pipeline = Pipeline(
    [
        ("selector", DataFrameSelector(category_attributes)),
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
        # ("label_binarizer", LabelBinarizer()),
        ("one_hot", OneHotEncoder(sparse=False)),
        # ("cat_encoder", CategorialEncoder(encoding="onehot-dense")),
    ]
)

full_pipeline = FeatureUnion(
    transformer_list=[
        ("numeric_pipeline", numeric_pipeline),
        ("category_pipeline", category_pipeline),
    ]
)

# test1 = category_pipeline.fit_transform(housing)
housing_prepared = full_pipeline.fit_transform(housing)


# # Run pipeline with dataset
# housing_num_transformed = numeric_pipeline.fit_transform(housing_num)

housing_prepared_dataframe = pd.DataFrame(
    housing_prepared,
    columns=itertools.chain.from_iterable(
        [
            numeric_attributes,
            [
                "rooms_per_household",
                "population_per_household",
                "add_bedrooms_per_room",
            ],
            category_pipeline.named_steps["one_hot"].get_feature_names(),
        ]
    ),
)
