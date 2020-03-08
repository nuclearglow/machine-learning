#!/usr/bin/env python
import os
import pandas
import joblib
import matplotlib.pyplot as plt

# Pandas DataFrame docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

# load training data
training_data_raw = joblib.load("data/01_test_data_raw.pkl")

# check count of non-null values
training_data_raw.info()

# A simple scatter plot of districts
training_data_raw.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# A fancier versio with district population marker-size-coded and
# with house value color-coded
training_data_raw.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=training_data_raw["population"] / 100,
    label="population",
    figsize=(20, 12),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.legend()

# Compute a correlation matrix
corr_matrix = training_data_raw.corr()
