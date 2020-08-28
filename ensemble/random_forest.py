#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import joblib
import multiprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from colors import cga_basic

# Train random forest on iris
iris = load_iris()
iris_random_forest_clf = RandomForestClassifier(
    n_estimators=500, n_jobs=multiprocessing.cpu_count() - 2
)
iris_random_forest_clf.fit(iris["data"], iris["target"])

# how important is each feature ?
for name, score in zip(
    iris["feature_names"], iris_random_forest_clf.feature_importances_
):
    print(f"feature: {name} - score: {score}")

# Load MNIST dataset
mnist_X_train = joblib.load("data/mnist_training_data.pkl")
mnist_y_train = joblib.load("data/mnist_training_labels.pkl")

mnist_X_test = joblib.load("data/mnist_test_data.pkl")
mnist_y_test = joblib.load("data/mnist_test_labels.pkl")

# Train random forest on MNIST dat
mnist_random_forest_clf = RandomForestClassifier(
    n_estimators=500, n_jobs=multiprocessing.cpu_count() - 2
)
mnist_random_forest_clf.fit(mnist_X_train, mnist_y_train)

# Extract feature importance of image data
feature_importance_pixels = mnist_random_forest_clf.feature_importances_.reshape(28, 28)

# Plot
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_title(f"MNIST Pixel Importance")
cf = ax.imshow(
    feature_importance_pixels,
    cmap=cga_basic,
    vmin=0,
    vmax=feature_importance_pixels.max(),
)
fig.colorbar(
    cf,
    boundaries=np.linspace(0, feature_importance_pixels.max(), 256),
    orientation="vertical",
)
