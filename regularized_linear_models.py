#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing

from scipy.interpolate import interp1d

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, SGDRegressor

from util import plot_learning_curve


# training_data
m = 100
# x = Vec(100,1) between -3 and 3
X = 6 * np.random.rand(m, 1) - 3
# y = f(x) = 1/2x^2 + x + 2 + random(100,1) between 0.5 and 10.5
y = 0.5 * X ** 2 + X + 2 + np.random.rand(m, 1)

# Plot training data
plt.scatter(X, y, c="#AA00AA")
plt.show()

ridge_regression = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=1, solver="cholesky")),
    ]
)

# now the augmented dataset with the polynomial expansion (**2) can be fitted
ridge_reg = ridge_regression.fit(X, y)
ridge_y_hat = ridge_reg.predict(X)

# Interpolate values
n_data_points = 500
x_new = np.linspace(X.min(), X.max(), n_data_points)
f = interp1d(X.ravel(), ridge_y_hat, kind="quadratic", axis=0)
y_smooth = f(x_new)

# Plot values versus predicted values
plt.plot(x_new, y_smooth, linestyle="-", color="#AA00AA")
plt.scatter(X, y, c="#00AAAA")
plt.scatter(X, ridge_y_hat, c="#FFFF55")
plt.axis([-3, 3, 0, 10])
plt.show()

# Plot learning curve
plot_learning_curve(
    ridge_regression,
    X,
    y,
    train_sizes=np.linspace(0.1, 1, 30),
    cv=5,
    n_jobs=multiprocessing.cpu_count() - 2,
    scoring="explained_variance",
)

# Ridge regression using SGDregression
sgd_regression_l2 = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        (
            "sgd_regression_l2",
            SGDRegressor(
                max_iter=1000, alpha=1e-5, penalty="l2", eta0=0.1, random_state=42
            ),
        ),
    ]
)

sgd_regression_l2.fit(X, y.ravel())
sgd_y_hat = sgd_regression_l2.predict(X)

# Interpolate values
n_data_points = 500
x_new = np.linspace(X.min(), X.max(), n_data_points)
f = interp1d(X.ravel(), sgd_y_hat, kind="quadratic", axis=0)
y_smooth = f(x_new)

# Plot values versus predicted values
plt.plot(x_new, y_smooth, linestyle="-", color="#AA00AA")
plt.scatter(X, y, c="#00AAAA")
plt.scatter(X, sgd_y_hat, c="#FFFF55")
# plt.axis([-3, 3, 0, 10])
plt.show()

# Plot learning curve
plot_learning_curve(
    sgd_regression_l2,
    X,
    y,
    train_sizes=np.linspace(0.1, 1, 30),
    cv=5,
    n_jobs=multiprocessing.cpu_count() - 2,
    scoring="explained_variance",
)
