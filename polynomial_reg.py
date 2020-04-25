#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from scipy.interpolate import interp1d


# training_data
m = 100
# x = Vec(100,1) between -3 and 3
X = 6 * np.random.rand(m, 1) - 3
# y = f(x) = 1/2x^2 + x + 2 + random(100,1) between 0.5 and 10.5
y = 0.5 * X ** 2 + X + 2 + np.random.rand(m, 1)


plt.scatter(X, y, c="#AA00AA")
plt.show()

polynomial_regression = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ]
)

# now the augmented dataset with the polynomial expansion (**2) can be fitted
lin_reg = polynomial_regression.fit(X, y)
y_hat = lin_reg.predict(X)

# Interpolate values
n_data_points = 500
x_new = np.linspace(X.min(), X.max(), n_data_points)
f = interp1d(X.ravel(), y_hat, kind="quadratic", axis=0)
y_smooth = f(x_new)

# Plot values versus predicted values
plt.plot(x_new, y_smooth, linestyle="-", color="#AA00AA")
plt.scatter(X, y, c="#00AAAA")
plt.scatter(X, y_hat, c="#FFFF55")
plt.axis([-3, 3, 0, 10])
plt.show()


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="trainining data")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="validation")
    plt.axis([0, 80, 0, 3])


plot_learning_curves(polynomial_regression, X, y)

# y = 4 + 3 * X + np.random.randn(100, 1)

# # prepare X to inclide first Vector with ones
# X_b = np.c_[np.ones((100, 1)), X]

# # old lin reg
# theta_lin_reg = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# # Batch gradient descent

# # learning rate
# eta = 0.1
# n_iterations = 1000

# # random initialization of theta
# theta_gradient_descent = np.random.randn(2, 1)

# # theta_next = theta - learning_rate * MSE_theta
# for iteration in range(n_iterations):
#     gradients = 2 / m * X_b.T.dot(X_b.dot(theta_gradient_descent) - y)
#     theta_gradient_descent = theta_gradient_descent - eta * gradients

# # Stochastic Gradient Descent SGD

# n_epochs = 50
# t0, t1 = 5, 500  # Learning schedule hyperparameters


# def learning_schedule(t):
#     return t0 / (t + t1)


# # Theta tracking
# thetatrack = np.zeros((n_epochs * m, 2))

# theta_sgd = np.random.randn(2, 1)

# for epoch in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index : random_index + 1]
#         yi = y[random_index : random_index + 1]
#         gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
#         eta = learning_schedule(epoch * m + i)
#         theta_sgd = theta_sgd - eta * gradients
#         thetatrack[epoch * i + i, :] = theta_sgd.T

# # Plot tracked theta
# plt.scatter(thetatrack[:, 0], thetatrack[:, 1])
# plt.show()


# # same with sklaern
# sgd_reg = SGDRegressor(max_iter=1000, penalty=None, eta0=0.1, random_state=42)
# sgd_reg.fit(X, y.ravel())
