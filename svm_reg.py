#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

from mlxtend.plotting import plot_decision_regions


# Some data
m = 100
noise_std = 3
X = 10 * np.random.rand(m, 1)
y = 0.7 * X + np.random.randn(m, 1) * noise_std

# Fit 2 models
svm_reg1 = LinearSVR(epsilon=5, C=1)
svm_reg1.fit(X, y)

svm_reg2 = LinearSVR(epsilon=0.5, C=1)
svm_reg2.fit(X, y)

# Get regression lines
x_data1 = np.linspace(X.min(), X.max(), m)
y_data1 = svm_reg1.intercept_ + svm_reg1.coef_ * x_data1
x_data2 = np.linspace(X.min(), X.max(), m)
y_data2 = svm_reg2.intercept_ + svm_reg2.coef_ * x_data2

# A plot
fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].scatter(X, y, c="#FF557F")
axes[0].set_facecolor("#AAAAAA")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("A plot!")

axes[0].plot(x_data1, y_data1, color="#e042ff", linestyle="solid")
axes[0].plot(x_data1, y_data1 - svm_reg1.epsilon, color="#e042ff", linestyle="dashed")
axes[0].plot(x_data1, y_data1 + svm_reg1.epsilon, color="#e042ff", linestyle="dashed")

axes[1].scatter(X, y, c="#55FF55")
axes[1].set_facecolor("#AAAAAA")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("A second plot!")

axes[1].plot(x_data2, y_data2, color="#e042ff", linestyle="solid")
axes[1].plot(x_data2, y_data2 - svm_reg2.epsilon, color="#e042ff", linestyle="dashed")
axes[1].plot(x_data2, y_data2 + svm_reg2.epsilon, color="#e042ff", linestyle="dashed")


fig.set_facecolor("#11AAAA")
plt.tight_layout()


# Some data
m = 100
noise_std = 10
X = 10 * np.random.rand(m, 1)
y = 0.7 * X ** 2 + np.random.randn(m, 1) * noise_std

# Fit 2 models
svm_polyreg1 = SVR(kernel="poly", degree=2, epsilon=0.1, C=100)
svm_polyreg1.fit(X, y)

svm_polyreg2 = SVR(kernel="poly", degree=2, epsilon=0.1, C=0.01)
svm_polyreg2.fit(X, y)


x_data3 = np.linspace(X.min(), X.max(), m).reshape(m,1)
y_data3 = svm_polyreg1.predict(x_data3)

x_data4 = np.linspace(X.min(), X.max(), m).reshape(m,1)
y_data4 = svm_polyreg2.predict(x_data4)


# A plot
fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].scatter(X, y, c="#FF557F")
axes[0].set_facecolor("#AAAAAA")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("A plot!")

axes[0].plot(x_data3, y_data3, color="#e042ff", linestyle="solid")

axes[1].scatter(X, y, c="#55FF55")
axes[1].set_facecolor("#AAAAAA")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("A second plot!")

axes[1].plot(x_data4, y_data4, color="#e042ff", linestyle="solid")


fig.set_facecolor("#11AAAA")
plt.tight_layout()
