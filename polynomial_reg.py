#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline

from scipy.interpolate import interp1d

from util import plot_learning_curve

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
        ("scaler", StandardScaler()),
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


plot_learning_curve(
    polynomial_regression,
    X,
    y,
    train_sizes=np.linspace(0.1, 1, 50),
    cv=5,
    n_jobs=multiprocessing.cpu_count() - 2,
    scoring="neg_mean_squared_error",
)
