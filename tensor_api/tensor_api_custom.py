#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Helper libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
fashion_mnist = keras.datasets.fashion_mnist

# split data
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Reshape data
X_train_full = X_train_full.reshape(
    (X_train_full.shape[0], X_train_full.shape[1] * X_train_full.shape[2])
)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# Scaling
X_test = X_test / 255.0
X_train_full = X_train_full / 255.0

# split out validation set
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2
)


# Specify class names
class_names = np.array(
    [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
)

# Specify model
input_layer = keras.layers.Input(shape=X_train.shape[1], name="input")
hidden_layer_1 = keras.layers.Dense(300, activation="relu", name="hidden_layer_1")(
    input_layer
)
hidden_layer_2 = keras.layers.Dense(300, activation="relu", name="hidden_layer_2")(
    hidden_layer_1
)
output_layer = keras.layers.Dense(10, activation="softmax", name="output_layer")(
    hidden_layer_2
)
model = keras.Model(inputs=[input_layer], outputs=[output_layer])

# Print summmary
model.summary()

tf.config.experimental_run_functions_eagerly(True)

# custom loss function, proportion of wrong predictions
v1 = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
v2 = tf.constant(np.random.rand(10, 10))

tf.print("v1: ", v1, v1.shape)

# tf.print(v1)
# tf.print(v2)

# a1 = v1.numpy()
# a2 = v2.numpy()


def custom_loss(y_true, y_pred):

    y_pred = tf.cast(
        tf.expand_dims(tf.math.argmax(y_pred, axis=1, output_type=tf.int32), 1),
        dtype=tf.float32,
    )
    # tf.print("y_true: ", y_true, y_true.shape, y_true.dtype)
    # tf.print("y_pred: ", y_pred, y_pred.shape, y_pred.dtype)

    y_delta = tf.math.subtract(y_true, y_pred)
    # tf.print("y_delta: ", y_delta, y_delta.shape, y_delta.dtype)

    n_nonzero = tf.math.count_nonzero(y_delta, dtype=tf.float32)
    tf.print("n_nonzero: ", n_nonzero, n_nonzero.shape, n_nonzero.dtype)

    # tmp = tf.shape(y_true)[0]
    # tf.print("tmp: ", tmp, tmp.dtype)

    n_observations = tf.constant(y_true.shape[0])
    tf.print(
        "n_observations: ", n_observations,
    )

    prop_wrong = tf.math.divide(n_nonzero, n_observations)
    tf.print(
        "prop_wrong: ", prop_wrong,
    )

    # y_true1 = tf.cast(y_true, dtype=tf.int32)
    # delta = y_true1 - y_pred1
    # n_nonzero = tf.math.count_nonzero(delta)
    return prop_wrong


# y_pred, y_delta = custom_loss(v1, v2)

# print("Test", loss, output_stream=sys.stdout)
# compile with options
model.compile(
    # TODO: custom loss
    loss=custom_loss,
    # TODO: custom metrics
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.Nadam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"
    ),
)

# Path to the trained model
model_path = os.path.abspath(
    f"{os.getcwd()}/trained_models/{os.path.basename(__file__)}.h5"
)

# Define callbacks
cb_early_stop = keras.callbacks.EarlyStopping(patience=10, monitor="val_loss")
cb_checkpoint = keras.callbacks.ModelCheckpoint(model_path)

# train the AI model
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[cb_early_stop, cb_checkpoint],
)


# plot the history of the early stopping monitor
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# eval against test set
model_evaluation = model.evaluate(X_test, y_test, return_dict=True)

# predict class probability
# X_new = X_test[:3]
# y_proba = model.predict(X_new)
# y_proba.round(3)

# predict classes
# y_class_prediction = class_names[y_proba.argmax(axis=1).astype(np.int)]






