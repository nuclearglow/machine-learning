#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data, obviously...
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape X-data
X_train_full = X_train_full.reshape((X_train_full.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Scale X-data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.fit_transform(X_test)

# Split a validatio set
X_train, X_train_valid, y_train, y_train_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2
)

# Compile a model
# model = keras.models.Sequential()
# model.add(keras.layers.Input(shape=X_train.shape[1], name="input"))
# for layer in range(2):
#     model.add(keras.layers.Dense(500, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))
# optimizer = keras.optimizers.SGD(lr=0.01)
# model.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer=optimizer,
#     metrics=["accuracy"],
# )
# model.summary()

input_layer = keras.layers.Input(shape=X_train_full.shape[1], name="input")
hidden_1 = keras.layers.Dense(1000, activation="relu", name="hidden_1")(input_layer)
hidden_2 = keras.layers.Dense(1000, activation="relu", name="hidden_2")(hidden_1)
output_layer = keras.layers.Dense(10, activation="softmax", name="output")(hidden_2)
model = keras.Model(inputs=[input_layer], outputs=[output_layer])

# Compile model
model.summary()
model.compile(
    loss=["sparse_categorical_crossentropy"],
    optimizer=keras.optimizers.SGD(lr=0.01),
    metrics=["accuracy"],
)


# Train the model
model.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_data=(X_train_valid, y_train_valid),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
    ],
)

model.save("trained_models/mnist_best_model.h5")

# Load best model
model = tf.keras.models.load_model("trained_models/mnist_best_model.h5")

# Predict
y_predicted = model.predict(X_test)
predictions = y_predicted.argmax(axis=1).astype(np.int)

# Get model accuracy
model_accuracy = accuracy_score(y_test, predictions, normalize=True)
