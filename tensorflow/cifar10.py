#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = keras.datasets.cifar10.load_data()

training_data, test_data = keras.datasets.cifar10.load_data()
X_train_full, y_train_full = training_data
X_test, y_test = test_data

# Split data in stratified manner
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, stratify=y_train_full
)

# preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape((X_train.shape[0], -1)))
X_valid = scaler.fit_transform(X_valid.reshape((X_valid.shape[0], -1)))
X_test = scaler.fit_transform(X_test.reshape((X_test.shape[0], -1)))

# Determine number of classes
n_classes = np.unique(y_train_full).shape[0]

# Plot model history function
def plot_model_history(history):
    epochs = history.epoch
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    plt.title("History")
    plt.plot(epochs, accuracy)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(epochs, accuracy, label="accuracy")
    ax.plot(epochs, val_accuracy, label="val_accuracy")
    ax.set_title("model accuracy history")
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracies")
    ax.legend(loc=0)


# Function builds and returns a regression model
def build_model(
    n_hidden=20,
    n_neurons=100,
    input_shape=[3072],
    learning_rate=3e-4,
    batch_normalization=False,
    dropout_rate=0.2,
):
    # init
    model = keras.models.Sequential()

    # input (cherntime, lat, lng, cherndist)
    model.add(keras.layers.Input(shape=input_shape, name="input"))
    # optionally, use batch normalization
    if batch_normalization:
        model.add(keras.layers.BatchNormalization())
    # n hidden dense layer with n neurons each
    for layer in range(n_hidden):
        model.add(
            keras.layers.Dense(
                n_neurons, activation="selu", kernel_initializer="lecun_normal",
            )
        )
        # optionally, use batch normalization
        if batch_normalization:
            model.add(keras.layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(rate=dropout_rate))
    # output layer 1 neurone
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    # model optimizer, enhance with Momentum Optimization, and use Nesterov Accelerated Optimization (look ahead in theta)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, name="Nadam")
    # optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    model.summary()
    return model


cifar_model = build_model()

# Checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    f"{os.getcwd()}/trained_models/model_cifar10.h5", save_best_only=True
)

# Early stopping callback
earlystop_cb = keras.callbacks.EarlyStopping(monitor="loss", patience=20)

# Fit the new model
history = cifar_model.fit(
    X_train,
    y_train,
    epochs=10000,
    batch_size=128,
    validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb, earlystop_cb],
)

plot_model_history(history)
