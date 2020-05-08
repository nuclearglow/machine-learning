#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

# Some data
m = 100
noise_std = 1.2
X = 10 * np.random.rand(m, 1)
y = 0.7 * X + np.random.randn(m, 1) * noise_std

# A plot
fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].scatter(X, y, c="#FF557F")
axes[0].set_facecolor("#AAAAAA")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("A plot!")

axes[1].scatter(X, y, c="#55FF55")
axes[1].set_facecolor("#AAAAAA")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("A second plot!")

fig.set_facecolor("#11AAAA")
