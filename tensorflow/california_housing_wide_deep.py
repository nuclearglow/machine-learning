#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test = scaler.fit_transform(X_test)


input_wide = keras.layers.Input(shape=[5], name="input_wide")
input_deep = keras.layers.Input(shape=[6], name="input_deep")

hidden1 = keras.layers.Dense(30, activation="relu")(input_deep)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

concat = keras.layers.Concatenate()([input_wide, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_wide, input_deep], outputs=[output, aux_output])
model.summary()

model.compile(
    loss=["mean_squared_error", "mean_squared_error"],
    loss_weights=[0.9, 0.1],
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
)

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:10, :], X_test_deep[:10, :]

history = model.fit(
    [X_train_wide, X_train_deep],
    [y_train, y_train],
    epochs=20,
    validation_data=([X_valid_wide, X_valid_deep], [y_valid, y_valid]),
)

mse_test = model.evaluate([X_test_wide, X_test_deep], [y_test, y_test])

X_new = X_test[:3]
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
