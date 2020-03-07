#!/usr/bin/env python
import os
import tarfile
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import util
import joblib

# load test and trainign data
housing_test_data = joblib.load("models/housing_test_data.pkl")
housing_training_data = joblib.load("models/housing_training_data.pkl")

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

