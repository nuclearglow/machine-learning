#!/usr/bin/env python
import os
import tarfile
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import util
import joblib
from sklearn.metrics import mean_squared_error

# load test and training data
housing_test_prepared = joblib.load("models/housing_test_data.pkl")
housing_test_labels = joblib.load("models/housing_test_labels.pkl")

# load final model
final_model = joblib.load("models/housing_final_model.pkl")

# final predictions
final_predictions = final_model.predict(housing_test_prepared)

# RMSE
final_mse = mean_squared_error(housing_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
