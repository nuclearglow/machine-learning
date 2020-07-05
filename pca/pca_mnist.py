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
X_train = joblib.load("../mnist/data/mnist_training_data.pkl")
y_train = joblib.load("../mnist/data/mnist_training_labels.pkl")

# Reduce via PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

# the good, the bad, and the leftover crack
image_number = 666
original_image = X_train[image_number, :].reshape(28, 28)
recovered_image = X_recovered[image_number, :].reshape(28, 28)

pca2 = PCA(n_components=0.95)
X_reduced2 = pca.fit_transform(original_image)
X_recovered2 = pca.inverse_transform(X_reduced2)


# Plot images
plt.subplot(2, 2, 1)
plt.title("the good")
plt.imshow(original_image, cmap=cga_p1_light, interpolation="none")
plt.subplot(2, 2, 2)
plt.title("the leftover crack")
plt.imshow(recovered_image, cmap=cga_p1_light, interpolation="none")

plt.subplot(2, 2, 3)
plt.title("the good")
plt.imshow(original_image, cmap=cga_p1_light, interpolation="none")
plt.subplot(2, 2, 4)
plt.title("the leftover crack")
plt.imshow(X_recovered2, cmap=cga_p1_light, interpolation="none")