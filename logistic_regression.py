#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load iris dataset as bunch
iris_data = load_iris()

# Get petal width
X = iris_data["data"][:, 3:]

# Recode target vector: 1 if class = 2, else 0
y = (iris_data["target"] == 2).astype(np.int)

# Initialize logistic regression model
log_reg = LogisticRegression()

# Fit petal widths and
log_reg.fit(X, y)

score = log_reg.score(X, y)

# plot probabilities
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virgin")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virgin")
plt.legend()
plt.title("1 feature")
plt.show()

# Get petal width and length
X2 = iris_data["data"][:, 2:]

# Recode target vector: 1 if class = 2, else 0
y2 = (iris_data["target"] == 2).astype(np.int)

# Initialize logistic regression model
log_reg2 = LogisticRegression()

# Fit petal widths and
log_reg2.fit(X2, y2)

score2 = log_reg2.score(X2, y2)

# # plot probabilities
y2_prediction = log_reg2.predict(X2)

petal_width = X2[:, 1]
petal_length = X2[:, 0]


# Plot true positives
plt.scatter(
    petal_length[(y2_prediction == 1) & (y2 == 1)],
    petal_width[(y2_prediction == 1) & (y2 == 1)],
    c="green",
    marker="o",
    label="Iris-Virgin True Positives",
)
# Plot tfalserue positives
plt.scatter(
    petal_length[(y2_prediction == 1) & (y2 == 0)],
    petal_width[(y2_prediction == 1) & (y2 == 0)],
    c="red",
    marker="o",
    label="Iris-Virgin False Positives",
)
# Plot true negatives
plt.scatter(
    petal_length[(y2_prediction == 0) & (y2 == 0)],
    petal_width[(y2_prediction == 0) & (y2 == 0)],
    c="green",
    marker="^",
    label="Iris-Virgin True Negatives",
)
# Plot false negatives
plt.scatter(
    petal_length[(y2_prediction == 0) & (y2 == 1)],
    petal_width[(y2_prediction == 0) & (y2 == 1)],
    c="red",
    marker="^",
    label="Iris-Virgin False Negatives",
)


plt.ylim((0, 3))
plt.xlim((0, 7))
# plt.plot(X2, y_proba[:, 1], "g-", label="Iris-Virgin")
# plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virgin")
# plt.legend()
plt.title("2 features")
plt.legend()
plt.show()

# Softmax Regression

# Get petal width and length
X_data = iris_data["data"]

# Select features for the model
feats = [0, 1, 2, 3]
X_softmax = X_data[:, feats]

# 0,1,2 = Iris-Setosa, Iris-Versicolour, Iris-Virginica
y_softmax = iris_data["target"]

# Initialize logistic regression model
log_reg_softmax = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=0.99)

# Fit petal widths and
log_reg_softmax.fit(X_softmax, y_softmax)

softmax_prediction = log_reg_softmax.predict(X_softmax)
softmax_proba = log_reg_softmax.predict_proba(X_softmax)
softmax_score = log_reg_softmax.score(X_softmax, y_softmax)

# Dictionaries for features and classes
classes = {0: "Iris-Setosa", 1: "Iris-Versicolour", 2: "Iris-Virginica"}
features = {0: "Sepal length", 1: "Sepal width", 2: "Petal length", 3: "Petal width"}
symbols = ["o", "^", "*"]

feature1 = X_data[:, feats[0]]
feature2 = X_data[:, feats[1]]

for i, (k, plant) in enumerate(classes.items()):
    plt.scatter(
        feature1[(softmax_prediction == k) & (y_softmax == k)],
        feature2[(softmax_prediction == k) & (y_softmax == k)],
        c="green",
        marker=symbols[i],
        label=f"{plant} correct",
    )
    plt.scatter(
        feature1[(softmax_prediction == k) & (y_softmax != k)],
        feature2[(softmax_prediction == k) & (y_softmax != k)],
        c="red",
        marker=symbols[i],
        label=f"{plant} false",
    )

plt.ylabel(str(features[feats[1]]))
plt.xlabel(str(features[feats[0]]))
plt.legend()
plt.title("Iris softmax classification")
plt.xlim((0, feature1.max() * 1.1))
plt.ylim((0, feature2.max() * 1.1))
plt.show()
