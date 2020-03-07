#!/usr/bin/env python
import os
import tarfile
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import util
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# load test and training data
housing_test_data = joblib.load("models/housing_test_data.pkl")
housing_training_data = joblib.load("models/housing_training_data.pkl")

# load labels
housing_labels = joblib.load("models/housing_labels.pkl")

# Random Forest Regression
# Available Ensemble Methods: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
forest_reg = joblib.load("models/housing_random_forest_regression.pkl")

# Cross-Validation of the Models
# Available Model Selectors: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
# Available Model Validation: https://scikit-learn.org/stable/modules/classes.html#model-validation
# cross_val_score: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
# Documentation: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
# scoring parameter: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
random_forest_scores = joblib.load(
    "models/housing_random_forest_regression_cross_val_scores.pkl"
)
random_forest_rmse_scores = np.sqrt(-random_forest_scores)


def display_scores(rmse_scores, training_score):
    print("Training Model RMSE:", training_score)
    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("Standard deviation:", rmse_scores.std())


print("Random Forest Housing Model Loaded")
display_scores(random_forest_scores, random_forest_rmse_scores)

parameter_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(
    forest_reg, parameter_grid, cv=5, scoring="neg_mean_squared_error"
)
grid_search.fit(housing_training_data, housing_labels)

# get best params
grid_search_best_params = grid_search.best_params_
grid_search_best_model = grid_search.best_estimator_

joblib.dump(grid_search_best_model, "models/housing_final_model.pkl")

print("Random Forest Optimized Model - GridSearchCV - Cross Validation Results")
cross_validation_results = grid_search.cv_results_
for mean_score, params in zip(
    cross_validation_results["mean_test_score"], cross_validation_results["params"]
):
    # rmse
    print(np.sqrt(-mean_score), params)

# TODO
# print("Random Forest Optimized Model - RandomizedSearchCV - Cross Validation Results")
# random_search = RandomizedSearchCV(
#     forest_reg, parameter_grid, cv=5, scoring="neg_mean_squared_error"
# )
