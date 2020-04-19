#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor

# training_data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# prepare X to inclide first Vector with ones
X_b = np.c_[np.ones((100, 1)), X]

# old lin reg
theta_lin_reg = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Batch gradient descent

# learning rate
eta = 0.1
n_iterations = 1000
m = 100

# random initialization of theta
theta_gradient_descent = np.random.randn(2, 1)

# theta_next = theta - learning_rate * MSE_theta
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta_gradient_descent) - y)
    theta_gradient_descent = theta_gradient_descent - eta * gradients

# Stochastic Gradient Descent SGD

n_epochs = 50
t0, t1 = 5, 500  # Learning schedule hyperparameters


def learning_schedule(t):
    return t0 / (t + t1)


# Theta tracking
thetatrack = np.zeros((n_epochs * m, 2))

theta_sgd = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        eta = learning_schedule(epoch * m + i)
        theta_sgd = theta_sgd - eta * gradients
        thetatrack[epoch * i + i, :] = theta_sgd.T

# Plot tracked theta
plt.scatter(thetatrack[:, 0], thetatrack[:, 1])
plt.show()


# same with sklaern
sgd_reg = SGDRegressor(max_iter=1000, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
