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
X, y = make_moons(noise=0, random_state=42)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

gamma_set = [0.1, 2.5, 5]
C_set = [0.001, 1, 1000]

resolution = 1000
x_ax_vals = np.linspace(X[:, 0].min() * 1.1, X[:, 0].max() * 1.1, resolution)
y_ax_vals = np.linspace(X[:, 1].min() * 1.1, X[:, 1].max() * 1.1, resolution)

pixels = np.array(list(itertools.product(x_ax_vals, y_ax_vals)))

for i, gamma in enumerate(gamma_set):
    for j, C in enumerate(C_set):

        # Init SVC
        svm_clf_rbf = SVC(kernel="rbf", probability=True, gamma=gamma, C=C)
        svm_clf_rbf.fit(X, y)

        # Predict classes
        class_prediction = svm_clf_rbf.predict(X)

        # Calculate probabilities
        predictions = svm_clf_rbf.predict_proba(pixels)[:, 0].reshape(
            (resolution, resolution)
        )

        # Plot
        plt.subplot(3, 3, (i * len(C_set) + j) + 1)

        plt.contourf(
            x_ax_vals,
            y_ax_vals,
            predictions,
            255,
            cmap=plt.cm.jet,
            vmin=predictions.min(),
            vmax=predictions.max(),
        )
        plt.colorbar(boundaries=np.linspace(predictions.min(), predictions.max(), 255))
        plt.title(f"gamma: {gamma} - C: {C}")

plt.show()


for i, gamma in enumerate(gamma_set):
    for j, C in enumerate(C_set):

        # Init SVC
        svm_clf_rbf = SVC(kernel="rbf", probability=True, gamma=gamma, C=C)
        svm_clf_rbf.fit(X, y)

        # Predict classes
        class_prediction = svm_clf_rbf.predict(X)

        # Plot
        plt.subplot(3, 3, (i * len(C_set) + j) + 1)

        # Plot decision regions
        plot_decision_regions(X, y, clf=svm_clf_rbf, legend=2)

        # Calculate probabilities
        predictions = svm_clf_rbf.predict_proba(pixels)[:, 0].reshape(
            (resolution, resolution)
        )
        plt.title(f"gamma: {gamma} - C: {C}")

plt.show()
