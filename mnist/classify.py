#!/usr/bin/env python
import joblib
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib

training_data = joblib.load("data/mnist_training_data.pkl")
training_labels = joblib.load("data/mnist_training_labels.pkl")

test_data = joblib.load("data/mnist_test_data.pkl")
test_labels = joblib.load("data/mnist_test_labels.pkl")

# Simplify the problem: extract indices where the label states 5 and use only the data subset
training_data_57 = training_data[(training_labels == "5") | (training_labels == "7"), :]
training_labels_57 = training_labels[
    (training_labels == "5") | (training_labels == "7")
]

# use SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(training_data_57, training_labels_57)
prediction = sgd_clf.predict([test_data[2]])


# Pick an example image
one_image = test_data[2]
one_image_2d = one_image.reshape(28, 28)

# Plot exxample image
matplotlib.pyplot.imshow(
    one_image_2d, cmap=matplotlib.cm.binary, interpolation="nearest"
)
matplotlib.pyplot.axis("off")
matplotlib.pyplot.show()
