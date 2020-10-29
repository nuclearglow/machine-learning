#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import numpy as np
import os

from sklearn.utils import validation
from tensorflow import keras
import geopy.distance
import datetime
import multiprocessing
import itertools
from scipy.stats import loguniform
from pytz import timezone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

# current directory with data file
preprocessed_data_path = os.path.abspath(
    f"{os.getcwd()}/data/chernair-preprocessing.pkl"
)

# Read data
data = joblib.load(preprocessed_data_path)

X = data[["lat", "lng", "cherndist", "cherntime"]].to_numpy()
y = data["I-131_interpolated"].to_numpy()

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_train_valid, y_train, y_train_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2
)

# Function builds and returns a regression model
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[4]):
    model = keras.models.Sequential()
    # input (cherntime, lat, lng, cherndist)
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    # n hidden dense layer with n neurons each
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    # output layer 1 neurone
    model.add(keras.layers.Dense(1))
    # model optimizer
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


# use a wrapper to get a scikit learn RegressorModel for the Keras factory method
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# parameter range for randomized search
parameter_distribution = {
    "n_hidden": list(range(4)),
    "n_neurons": np.arange(1, 101),
    "learning_rate": loguniform(3e-4, 3e-2, 10),
}

randomized_search_cv = RandomizedSearchCV(
    keras_reg, parameter_distribution, n_iter=10, cv=3
)

randomized_search_cv.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_train_valid, y_train_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)

best_params = randomized_search_cv.cv_best_params_
best_score = randomized_search_cv.cv_best_score_

# TODOs
# 3 Regressions-Modelle:
#   cherntime, lat, lng, cherndist, etc. -> Schätzt die Isotop-Konzentration
