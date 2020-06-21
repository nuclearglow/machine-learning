#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from colors import cga_p1_dark
from util import plot_learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# training_data -> polynomial degree 2
m = 500
# x = Vec(100,1) between -3 and 3
X = 6 * np.random.rand(m, 1) - 3
# y = f(x) = 1/2x^2 + x + 2 + random(100,1) between 0.5 and 10.5
y = X ** 2 + np.random.rand(m, 1)

plt.scatter(X, y, cmap=cga_p1_dark)
plt.show()

# Fit a decision tree
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)

# Get residuals
prediction1 = tree_reg1.predict(X).reshape(m, -1)
y2 = y - prediction1

# Fit second tree on residuals
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)

# Get residuals again...
prediction2 = tree_reg2.predict(X).reshape(m, -1)
y3 = y2 - prediction2

# Yet another tree...
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)

prediction3 = tree_reg3.predict(X).reshape(m, -1)

# Combining predictions
prediction_final = sum(tree.predict(X) for tree in (tree_reg1, tree_reg2, tree_reg3))

# Interpolate values
n_data_points = m
x_new = np.linspace(X.min(), X.max(), n_data_points)
f = interp1d(X.ravel(), prediction1, kind="quadratic", axis=0)
y_smooth1 = f(x_new)

f = interp1d(X.ravel(), prediction2, kind="quadratic", axis=0)
y_smooth2 = f(x_new)
f = interp1d(X.ravel(), prediction3, kind="quadratic", axis=0)
y_smooth3 = f(x_new)
f = interp1d(X.ravel(), prediction_final, kind="quadratic", axis=0)
y_smooth_final = f(x_new)

# Get colors
colors = cga_p1_dark(np.linspace(0, 1, 4))

# Plot values versus predicted values
plt.plot(x_new, y_smooth1, linewidth=3, linestyle="-", color=colors[0])
plt.plot(x_new, y_smooth2, linewidth=3, linestyle="-", color=colors[1])
plt.plot(x_new, y_smooth3, linewidth=3, linestyle="-", color=colors[2])
plt.plot(x_new, y_smooth_final, linewidth=3, linestyle="-", color=colors[3])
plt.scatter(X, y, c="#00AAAA")
plt.axis([-3, 3, 0, 10])
plt.show()

# Get training and test data for gradient boosting regressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, learning_rate=0.1)
gbrt.fit(X_train, y_train.ravel())

# get the mse for the sum of the predictor at all stages
errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt.staged_predict(X_test)]
# get the best estimator
best_n_estimators = np.argmin(errors)

# example: directly train with the best estimator
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
gbrt_best.fit(X_train, y_train.ravel())


# implement eraly stopping -> when the prediciton has not improved for 5 steps
gbrt2 = GradientBoostingRegressor(max_depth=2, warm_start=True, n_estimators=1)

min_val_error = float("inf")
error_going_up = 0
for i, n_estimators in enumerate(range(1, 120)):
    gbrt2.n_estimators = n_estimators
    gbrt2.fit(X_train, y_train.ravel())
    y_pred = gbrt2.predict(X_test)
    val_error = mean_squared_error(y_test.ravel(), y_pred.ravel())

    print(val_error)

    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            print(i, n_estimators)
            break

# directly use early stopping:

gbrt3 = GradientBoostingRegressor(max_depth=2, n_estimators=120, n_iter_no_change=5)
