#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Load iris dataset as bunch
iris_data = load_iris()

X = iris_data["data"][:, (2, 3)]
y = (iris_data["target"] == 2).astype(np.float64)

# Linear SVM Pipeline
svm_clf = Pipeline(
    [("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=100, loss="hinge")),]
)

svm_clf.fit(X, y)

class_prediction = svm_clf.predict(X)

# Dictionaries for features and classes
classes = {0: "other", 1: "Iris-Virginica"}
features = {0: "Petal length", 1: "Petal width"}
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
plt.title("Iris SVM classification")
plt.xlim((0, feature1.max() * 1.1))
plt.ylim((0, feature2.max() * 1.1))
plt.show()
