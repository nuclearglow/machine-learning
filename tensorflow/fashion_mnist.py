#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# split out validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# SPecify class names
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

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.summary()
print(model.layers)
weights, biases = model.layers[1].get_weights()
print(np.std(weights))

# compile with options
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

# train the AI model
history = model.fit(
    X_train, y_train, batch_size=128, epochs=30, validation_data=(X_valid, y_valid)
)


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# eval against test set
model.evaluate(X_test, y_test)

# predict class probability
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(3)

# predict classes
y_class_prediction = class_names[y_proba.argmax(axis=1).astype(np.int)]

