#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import cProfile

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from mlxtend.plotting import plot_decision_regions

cp = cProfile.Profile()
cp.enable()

# Load iris dataset as bunch
X, y = make_moons(noise=0.20, random_state=42)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Init SVC
svm_clf = SVC(kernel="poly", probability=True)

# Hyperparameter optimization via grid search
parameter_grid = [
    {
        "degree": np.arange(1, 11),
        "coef0": np.arange(0, 10, 0.1),
        "C": np.arange(0.01, 0.2, 0.01),
    },
]

grid_search = GridSearchCV(
    svm_clf, parameter_grid, n_jobs=multiprocessing.cpu_count() - 2, cv=5
)
grid_search.fit(X, y)

grid_search_best_params = grid_search.best_params_
grid_search_best_model = grid_search.best_estimator_

class_prediction = grid_search_best_model.predict(X)


plot_decision_regions(X, y, clf=grid_search_best_model, legend=2)

# Dictionaries for features and classes
classes = {0: "class 0", 1: "class 1"}
features = {0: "feature 1", 1: "feature 2"}
symbols = ["o", "*"]

feature1 = X[:, 0]
feature2 = X[:, 1]

for i, (k, plant) in enumerate(classes.items()):
    plt.scatter(
        feature1[(class_prediction == k) & (y == k)],
        feature2[(class_prediction == k) & (y == k)],
        c="green",
        marker=symbols[i],
        label=f"{plant} correct",
    )
    plt.scatter(
        feature1[(class_prediction == k) & (y != k)],
        feature2[(class_prediction == k) & (y != k)],
        c="red",
        marker=symbols[i],
        label=f"{plant} false",
    )

plt.ylabel(str(features[1]))
plt.xlabel(str(features[0]))
plt.legend()
plt.title("moons SVM classification")
plt.xlim(feature1.min() * 1.1, feature1.max() * 1.1)
plt.ylim(feature2.min() * 1.1, feature2.max() * 1.1)
plt.show()


resolution = 1000
x_ax_vals = np.linspace(X[:, 0].min() * 1.1, X[:, 0].max() * 1.1, resolution)
y_ax_vals = np.linspace(X[:, 1].min() * 1.1, X[:, 1].max() * 1.1, resolution)

pixels = np.array(list(itertools.product(x_ax_vals, y_ax_vals)))
predictions = grid_search_best_model.predict_proba(pixels)[:, 0].reshape(
    (resolution, resolution)
)

# plt.contour(x_ax_vals, y_ax_vals, pixels, 50, linewidths=0.5, colors='k')
# detect dpi https://www.infobyip.com/detectmonitordpi.php
my_dpi = 96
plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
plt.contourf(
    x_ax_vals, y_ax_vals, predictions, 255, cmap=plt.cm.hsv, vmin=0, vmax=1
)
plt.colorbar()
plt.show()

cp.disable()
cp.print_stats()
