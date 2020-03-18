#!/usr/bin/env python
from sklearn.datasets import fetch_openml
import joblib
import numpy as np

# use sklearn download functionality
mnist_data = fetch_openml("mnist_784")

# extract our stuff
images, labels = mnist_data["data"], mnist_data["target"]

# data is pre-split at datapoint 60000, use that
training_data, test_data = images[:60000], images[60000:]
training_labels, test_labels = labels[:60000], labels[60000:]

# Shuffle training data and labels
shuffle_index = np.random.permutation(training_data.shape[0])
training_data, training_labels = (
    training_data[shuffle_index],
    training_labels[shuffle_index],
)

# save everything
joblib.dump(mnist_data, "data/mnist_data_raw.pkl")

joblib.dump(training_data, "data/mnist_training_data.pkl")
joblib.dump(test_data, "data/mnist_test_data.pkl")
joblib.dump(training_labels, "data/mnist_training_labels.pkl")
joblib.dump(test_labels, "data/mnist_test_labels.pkl")
