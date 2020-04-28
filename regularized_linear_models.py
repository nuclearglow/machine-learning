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
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

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
        ("ridge_reg", Ridge(alpha=0.1, solver="cholesky")),
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
    title="Ridge regression learning curve",
)

# Ridge regression using SGDregression
sgd_regression_l2 = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        (
            "sgd_regression_l2",
            SGDRegressor(
                max_iter=1000, alpha=0.1, penalty="l2", eta0=0.1, random_state=42
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
    title="Ridge regression via SGD learning curve",
)

# L.A.S.S.O. = Least Absolute Shrinkage and Selection Operator Regression

# LASSO Regression using SGDregression
sgd_regression_l1 = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        (
            "sgd_regression_l1",
            SGDRegressor(
                max_iter=1000, alpha=0.1, penalty="l1", eta0=0.1, random_state=42
            ),
        ),
    ]
)

sgd_regression_l1.fit(X, y.ravel())
sgd_y_hat_lasso = sgd_regression_l1.predict(X)

# Interpolate values
n_data_points = 500
x_new = np.linspace(X.min(), X.max(), n_data_points)
f = interp1d(X.ravel(), sgd_y_hat_lasso, kind="quadratic", axis=0)
y_smooth = f(x_new)

# Plot values versus predicted values
plt.plot(x_new, y_smooth, linestyle="-", color="#AA00AA")
plt.scatter(X, y, c="#00AAAA")
plt.scatter(X, sgd_y_hat_lasso, c="#FFFF55")
# plt.axis([-3, 3, 0, 10])
plt.show()

# Plot learning curve
plot_learning_curve(
    sgd_regression_l1,
    X,
    y,
    train_sizes=np.linspace(0.1, 1, 30),
    cv=5,
    n_jobs=multiprocessing.cpu_count() - 2,
    scoring="explained_variance",
    title="Lasso regression via SGD learning curve",
)

# LASSO Regression using Polynomial Features and Linear Regression

lasso_regression = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=0.1)),
    ]
)

# now the augmented dataset with the polynomial expansion (**2) can be fitted
lasso_reg = lasso_regression.fit(X, y)
lasso_y_hat = lasso_reg.predict(X)

# Interpolate values
n_data_points = 500
x_new = np.linspace(X.min(), X.max(), n_data_points)
f = interp1d(X.ravel(), lasso_y_hat, kind="quadratic", axis=0)
y_smooth = f(x_new)

# Plot values versus predicted values
plt.plot(x_new, y_smooth, linestyle="-", color="#AA00AA")
plt.scatter(X, y, c="#00AAAA")
plt.scatter(X, lasso_y_hat, c="#FFFF55")
plt.axis([-3, 3, 0, 10])
plt.show()

# Plot learning curve
plot_learning_curve(
    lasso_regression,
    X,
    y,
    train_sizes=np.linspace(0.1, 1, 30),
    cv=5,
    n_jobs=multiprocessing.cpu_count() - 2,
    scoring="explained_variance",
    title="Lasso regression learning curve",
)

# Elastic Net Regression
# r=0 Ridge-Regression, r=1 = Lasso-Regression

elastic_net_regression = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("elastic_net_reg", ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ]
)

# now the augmented dataset with the polynomial expansion (**2) can be fitted
elastic_net_reg = elastic_net_regression.fit(X, y)
elastic_net_y_hat = elastic_net_reg.predict(X)

# Interpolate values
n_data_points = 500
x_new = np.linspace(X.min(), X.max(), n_data_points)
f = interp1d(X.ravel(), elastic_net_y_hat, kind="quadratic", axis=0)
y_smooth = f(x_new)

# Plot values versus predicted values
plt.plot(x_new, y_smooth, linestyle="-", color="#AA00AA")
plt.scatter(X, y, c="#00AAAA")
plt.scatter(X, elastic_net_y_hat, c="#FFFF55")
plt.axis([-3, 3, 0, 10])
plt.show()

# Plot learning curve
plot_learning_curve(
    elastic_net_regression,
    X,
    y,
    train_sizes=np.linspace(0.1, 1, 30),
    cv=5,
    n_jobs=multiprocessing.cpu_count() - 2,
    scoring="explained_variance",
    title="Elastic Net regression learning curve",
)

# Early stopping regularization of Regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

poly_scaler_pipeline = Pipeline(
    [
        ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
    ]
)

X_train_poly_scaled = poly_scaler_pipeline.fit_transform(X_train)
X_test_poly_scaled = poly_scaler_pipeline.fit_transform(X_test)

sgd_regressor = SGDRegressor(
    max_iter=1,
    warm_start=True,
    penalty=None,
    learning_rate="constant",
    eta0=0.0005,
    early_stopping=False,
)

minimum_mse = float("inf")
best_epoch = None
best_model = None

for epoch in range(1000):
    sgd_regressor.fit(X_train_poly_scaled, y_train)
    y_hat_test = sgd_regressor.predict(X_test_poly_scaled)
    y_test_mse = mean_squared_error(y_hat_test, y_test)

    if y_test_mse < minimum_mse:
        minimum_mse = y_test_mse
        best_epoch = epoch
        best_model = clone(sgd_regressor)

# directly use in sklearn
sgd_regressor_early_stop = SGDRegressor(
    max_iter=1000,
    warm_start=True,
    penalty=None,
    learning_rate="constant",
    eta0=0.0005,
    early_stopping=True,
    tol=1e-3,
    n_iter_no_change=5,
    verbose=1000,
)
sgd_regressor_early_stop.fit(X_train_poly_scaled, y_train)
