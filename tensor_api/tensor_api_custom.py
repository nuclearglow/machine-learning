#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Helper libraries
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

# compile with options
model.compile(
    # TODO: custom loss
    loss="sparse_categorical_crossentropy",
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

