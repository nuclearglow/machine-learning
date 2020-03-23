#!/usr/bin/env python
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib

training_data = joblib.load("data/mnist_training_data.pkl")
training_labels = joblib.load("data/mnist_training_labels.pkl")

test_data = joblib.load("data/mnist_test_data.pkl")
test_labels = joblib.load("data/mnist_test_labels.pkl")

# Simplify the problem: extract indices where the label states 5 and use only the data subset
# training_data_57 = training_data[(training_labels == "5") | (training_labels == "7"), :]
# training_labels_57 = training_labels[
#     (training_labels == "5") | (training_labels == "7")
# ]
# training_data[training_labels != "5"] = 10
# training_data_5 = training_data[training_labels != "5"]
# training_labels[training_labels != "5"] = "not5"
# training_labels_5 = training_labels[training_labels != "5"]

# Binarize labels, 5 and !5
training_labels_5 = training_labels.copy()
training_labels_5[training_labels_5 != "5"] = "not5"

# use SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(training_data_57, training_labels_57)
# prediction = sgd_clf.predict([test_data[2]])

# Cross validation of classifier (Stochastic Gradient Descent, SGD)
# accuracy_scores = cross_val_score(
#     sgd_clf, training_data, training_labels_5, cv=3, scoring="accuracy"
# )

# Cross Validation Prediction
training_label_predictions = cross_val_predict(
    sgd_clf, training_data, training_labels, cv=3
)
cm = confusion_matrix(
    training_labels, training_label_predictions, labels=np.unique(training_labels)
)

precision = precision_score(
    training_labels, training_label_predictions, average="weighted"
)
recall = recall_score(training_labels, training_label_predictions, average="weighted")

# F1 score is TP / (TP + ( (FN+FP) / 2 ))
f1 = f1_score(training_labels, training_label_predictions, average="weighted")

# Pick an example image
# one_image = test_data[2]
# one_image_2d = one_image.reshape(28, 28)

# Plot example image
# matplotlib.pyplot.imshow(
#     one_image_2d, cmap=matplotlib.cm.binary, interpolation="nearest"
# )
# matplotlib.pyplot.axis("off")
# matplotlib.pyplot.show()

