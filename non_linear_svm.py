#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC

from mlxtend.plotting import plot_decision_regions


# Load iris dataset as bunch
X, y = make_moons(noise=0.25, random_state=42)

# Linear SVM Pipeline
polynomial_svm_clf = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge")),
    ]
)

polynomial_svm_clf.fit(X, y)

class_prediction = polynomial_svm_clf.predict(X)

plot_decision_regions(X, y, clf=polynomial_svm_clf, legend=2)

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
