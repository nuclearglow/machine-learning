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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from pandas.plotting import scatter_matrix
from transformers.CombinedAttributesAdder import CombinedAttributesAdder
from transformers.DataFrameSelector import DataFrameSelector
import joblib

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

# labels = median house value
housing_labels = strat_train_set["median_house_value"].copy()

# Data cleanup: remove data which used to train/label against later
housing = strat_train_set.drop("median_house_value", axis=1)

category_attributes = ["ocean_proximity"]
numeric_attributes = util.difference(list(housing), category_attributes)

# Define our pipeline
# numeric part
numeric_pipeline = Pipeline(
    [
        ("selector", DataFrameSelector(numeric_attributes)),
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)
# categorial part
category_pipeline = Pipeline(
    [
        ("selector", DataFrameSelector(category_attributes)),
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
        # ("label_binarizer", LabelBinarizer()),
        ("one_hot", OneHotEncoder(sparse=False)),
        # ("cat_encoder", CategorialEncoder(encoding="onehot-dense")),
    ]
)
# combined parallel pipeline
full_pipeline = FeatureUnion(
    transformer_list=[
        ("numeric_pipeline", numeric_pipeline),
        ("category_pipeline", category_pipeline),
    ]
)

# test1 = category_pipeline.fit_transform(housing)
housing_prepared = full_pipeline.fit_transform(housing)

# dataframe for analysis
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

# Select and train a model :-)


# Available Linear Models: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Available regression metrics: https://scikit-learn.org/stable/modules/classes.html#regression-metrics
# RMSE
linear_predictions = lin_reg.predict(housing_prepared)
linear_regression_mse = mean_squared_error(housing_labels, linear_predictions)
linear_regression_rmse = np.sqrt(linear_regression_mse)

# Decision Tree Model
# Avilable Decision Trees: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
decision_tree_predictions = tree_reg.predict(housing_prepared)
decision_tree_mse = mean_squared_error(housing_labels, decision_tree_predictions)
decision_tree_rmse = np.sqrt(decision_tree_mse)

# Random Forest Regression
# Available Ensemble Methods: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_predictions = forest_reg.predict(housing_prepared)
forest_tree_mse = mean_squared_error(housing_labels, forest_predictions)
forest_tree_rmse = np.sqrt(forest_tree_mse)

# Cross-Validation of the Models
# Available Model Seclectors: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
# Available Model Validation: https://scikit-learn.org/stable/modules/classes.html#model-validation
# cross_val_score: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
# Documentation: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
# scoring parameter: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
tree_reg_scores = cross_val_score(
    tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10
)
tree_rmse_scores = np.sqrt(-tree_reg_scores)

lin_reg_scores = cross_val_score(
    lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10
)
lin_reg_rmse_scores = np.sqrt(-lin_reg_scores)

random_forest_scores = cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10,
)
random_forest_rmse_scores = np.sqrt(-random_forest_scores)


def display_scores(rmse_scores, training_score):
    print("Training Model RMSE:", training_score)
    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("Standard deviation:", rmse_scores.std())


print("Linear")
display_scores(lin_reg_rmse_scores, linear_regression_rmse)
print("Tree")
display_scores(tree_rmse_scores, decision_tree_rmse)
print("Random")
display_scores(random_forest_rmse_scores, forest_tree_rmse)

# export models
joblib.dump(lin_reg, "models/housing_linear_regression.pkl")
joblib.dump(lin_reg_scores, "models/housing_linear_regression_cross_val_scores.pkl")
joblib.dump(forest_reg, "models/housing_random_forest_regression.pkl")
joblib.dump(
    random_forest_scores, "models/housing_random_forest_regression_cross_val_scores.pkl"
)
