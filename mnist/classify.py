#!/usr/bin/env python
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
training_labels_5[training_labels == "5"] = 1
training_labels_5[training_labels != "5"] = 0
training_labels_5 = np.array(training_labels_5, dtype=np.float64)

# use SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(training_data_57, training_labels_57)
# prediction = sgd_clf.predict([test_data[2]])

# Cross validation of classifier (Stochastic Gradient Descent, SGD)
# accuracy_scores = cross_val_score(
#     sgd_clf, training_data, training_labels_5, cv=3, scoring="accuracy"
# )

# Cross Validation Prediction
# training_label_predictions = cross_val_predict(
#    sgd_clf, training_data, training_labels_5, cv=3
# )
# cm = confusion_matrix(
#     training_labels_5, training_label_predictions, labels=np.unique(training_labels)
# )

# # Relevanz, Precision: TP / (TP + FP)
# precision = precision_score(
#     training_labels_5, training_label_predictions, average="weighted"
# )
# # Sensitivität, Trefferquote: TP / (TP + FN)
# recall = recall_score(training_labels_5, training_label_predictions, average="weighted")

# # F1 score is TP / (TP + ( (FN+FP) / 2 )) -> Gleicher Einfluss durch precision und recall
# f1 = f1_score(training_labels_5, training_label_predictions, average="weighted")

# use cross val predicion with decision function
decision_scores = cross_val_predict(
    sgd_clf, training_data, training_labels_5, cv=3, method="decision_function"
)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.title("Precision / Recall")
    plt.ylim([0, 1])


precisions, recalls, thresholds = precision_recall_curve(
    training_labels_5, decision_scores, pos_label=1
)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

thresh = -20000
prediction = np.zeros(decision_scores.shape)
prediction[decision_scores > thresh] = 1

precision = precision_score(training_labels_5, prediction)
recall = recall_score(training_labels_5, prediction)

# Relevanz, Precision: TP / (TP + FP)

# True Positive Rate = Recall (hit / hit + omission)
# TPR = TP / (TP + FN)

# False Positive Rate (false alarm / false alarm + correct rejection)
# FPR = FP / (FP + TN)

# Receiver Operator Characteristic = True Positive Rate auf Y / False Positive Rate auf X
fpr, tpr, thresholds = roc_curve(training_labels_5, decision_scores)

# Area under curve ROC_AUC
auc_score = roc_auc_score(training_labels_5, decision_scores)

# Interpret ROC: Diagonalen = Zufall,
def plot_roc_curve(fpr, tpr, title=None):
    plt.title(title)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


plot_roc_curve(fpr, tpr, title=f"ROC AUC = {auc_score:.4f}")
plt.show()

# Pick an example image
# one_image = test_data[2]
# one_image_2d = one_image.reshape(28, 28)

# Plot example image
# matplotlib.pyplot.imshow(
#     one_image_2d, cmap=matplotlib.cm.binary, interpolation="nearest"
# )
# matplotlib.pyplot.axis("off")
# matplotlib.pyplot.show()
