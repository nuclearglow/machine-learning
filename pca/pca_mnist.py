#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

# insert path to color module
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(1, module_path)
from util.colors import cga_p1_light

# Load MNIST dataset
X_train = joblib.load("data/mnist_training_data.pkl")
y_train = joblib.load("data/mnist_training_labels.pkl")

# Reduce via PCA
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

# the good, the bad, and the leftover crack
original_image = X_train[10, :].reshape(28, 28)
#reduced_image = X_reduced[10, :].reshape(28, 28)
recovered_image = X_recovered[10, :].reshape(28, 28)

# Plot images
plt.subplot(1, 3, 1)
plt.title("the good")
plt.imshow(original_image, cmap=matplotlib.cm.binary, interpolation="none")
plt.subplot(1, 3, 3)
plt.title("the leftover crack")
plt.imshow(recovered_image, cmap=matplotlib.cm.binary, interpolation="none")
