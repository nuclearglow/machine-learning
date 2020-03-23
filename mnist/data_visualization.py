#!/usr/bin/env python
import os
import pandas
import joblib
import matplotlib
import numpy as np

# Pandas DataFrame docs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

# load training data
digits_data = joblib.load("data/mnist_data_raw.pkl")

images, labels = digits_data["data"], digits_data["target"]

# Pick an example image
one_image = images[37564]
one_image_2d = one_image.reshape(28, 28)

# Plot exxample image
matplotlib.pyplot.imshow(
    one_image_2d, cmap=matplotlib.cm.binary, interpolation="nearest"
)
matplotlib.pyplot.axis("off")
matplotlib.pyplot.show()
