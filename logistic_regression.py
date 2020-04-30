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


# color_vector = y2
# color_vector[(color_vector == 0)] = "g"
# color_vector[(color_vector == 0)] = "r"

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
