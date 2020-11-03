#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import numpy as np
import os
import geopy.distance
import datetime
import multiprocessing
import itertools
from tensorflow import keras
from pytz import timezone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from pathlib import Path
import joblib

# current directory with data file
preprocessed_data_path = os.path.abspath(
    f"{os.getcwd()}/data/chernair-preprocessed.pkl"
)

# Read data
data = joblib.load(preprocessed_data_path)


# 1 Klassifikator-Modell
#   cherntime, isotop-konzentration ->Schätzt, welche Stadt
X = data[
    ["cherntime", "I-131_interpolated", "Cs-134_interpolated", "Cs-137_interpolated"]
].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = data["city_code"].to_numpy()

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_train_valid, y_train, y_train_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2
)

n_cities = np.unique(y).shape[0]

# Function builds and returns a regression model
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[4]):
    model = keras.models.Sequential()
    # input (cherntime, lat, lng, cherndist)
    model.add(keras.layers.Input(shape=input_shape, name="input"))
    # n hidden dense layer with n neurons each
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    # output layer 1 neurone
    model.add(keras.layers.Dense(n_cities, activation="softmax"))
    # model optimizer
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    model.summary()
    return model


# use a wrapper to get a scikit learn RegressorModel for the Keras factory method
keras_clf = keras.wrappers.scikit_learn.KerasClassifier(
    build_fn=build_model, nb_epoch=100, batch_size=32, verbose=True
)

# parameter range for randomized search
parameter_distribution = {
    "n_hidden": list(range(6)),
    "n_neurons": [5, 10, 20, 50, 100, 200, 500, 1000],
    "learning_rate": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
    "input_shape": [X_train.shape[1]],
}

grid_search_cv = GridSearchCV(keras_clf, parameter_distribution, cv=3)

grid_search_cv.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_train_valid, y_train_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)

best_params = grid_search_cv.best_params_
best_score = grid_search_cv.best_score_

optimized_model = build_model(
    n_hidden=best_params["n_hidden"],
    n_neurons=best_params["n_neurons"],
    learning_rate=best_params["learning_rate"],
)

# simple
optimized_model.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_data=(X_train_valid, y_train_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)

optimized_model.save("trained_models/model_predict_city_code.h5")
