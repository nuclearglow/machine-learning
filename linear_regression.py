#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# parameter_vector = np.array([[5, 7, 3, 2, 9, 4, 1, 8, 6]]).transpose()
# parameter_vector = np.concatenate(
#     (np.ones(parameter_vector.shape), parameter_vector), axis=1
# )

# feature_vector = np.array([[1, 2, 3, 4, 5, 7, 8, 6, 9]])

# y_hat = np.dot(feature_vector, parameter_vector)

# training_data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# plt.scatter(X, y)
# plt.ylim([0, 15])

# minimize theta to predict y
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [0.7], [2]])
X_new_b = np.c_[np.ones(X_new.shape), X_new]

y_predict = X_new_b.dot(theta_best)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# do the same with SciKit learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict_sklearn = lin_reg.predict(X_new)
