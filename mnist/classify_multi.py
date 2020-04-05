#!/usr/bin/env python
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


training_data = joblib.load("data/mnist_training_data.pkl")
training_labels = joblib.load("data/mnist_training_labels.pkl")

test_data = joblib.load("data/mnist_test_data.pkl")
test_labels = joblib.load("data/mnist_test_labels.pkl")


training_labels = np.array(training_labels, dtype=np.float64)

# use SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(training_data, training_labels)
# test
test_score = sgd_clf.decision_function([training_data[2, :]])
classes = sgd_clf.classes_
detected_class = sgd_clf.classes_[test_score.argmax()]
# scale the data
scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_data)
# cross validation
sgd_cross_validation_score = cross_val_score(
    sgd_clf, training_data_scaled, training_labels, cv=3, scoring="accuracy"
)
# cross validation prediction scores
training_label_predictions = cross_val_predict(
    sgd_clf, training_data_scaled, training_labels, cv=3
)
confmat = confusion_matrix(training_labels, training_label_predictions)

# Error Analysis
conf_mat_row_sums = confmat.sum(axis=1, keepdims=True)
normalized_confusion_matrix = confmat / conf_mat_row_sums
np.fill_diagonal(normalized_confusion_matrix, 0)

# Plot confusion matrix
plt.matshow(normalized_confusion_matrix, cmap=plt.cm.jet)
plt.show()

# altermative classifier: use random forest classifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(training_data, training_labels)
# test
forest_probability_scores = forest_clf.predict_proba([training_data[0, :]])
# Cross validation
forest_cross_validation_score = cross_val_score(
    forest_clf, training_data, training_labels, cv=3, scoring="accuracy"
)


# # Cross validation of classifier (Stochastic Gradient Descent, SGD)
# # accuracy_scores = cross_val_score(
# #     sgd_clf, training_data, training_labels_5, cv=3, scoring="accuracy"
# # )

# # Cross Validation Prediction
# # training_label_predictions = cross_val_predict(
# #    sgd_clf, training_data, training_labels_5, cv=3
# # )
# # cm = confusion_matrix(
# #     training_labels_5, training_label_predictions, labels=np.unique(training_labels)
# # )

# # # Relevanz, Precision: TP / (TP + FP)
# # precision = precision_score(
# #     training_labels_5, training_label_predictions, average="weighted"
# # )
# # # Sensitivität, Trefferquote: TP / (TP + FN)
# # recall = recall_score(training_labels_5, training_label_predictions, average="weighted")

# # # F1 score is TP / (TP + ( (FN+FP) / 2 )) -> Gleicher Einfluss durch precision und recall
# # f1 = f1_score(training_labels_5, training_label_predictions, average="weighted")

# # use cross val predicion with decision function
# decision_scores = cross_val_predict(
#     sgd_clf, training_data, training_labels_5, cv=3, method="decision_function"
# )


# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
#     plt.xlabel("Threshold")
#     plt.legend(loc="center left")
#     plt.title("Precision / Recall")
#     plt.ylim([0, 1])


# precisions, recalls, thresholds = precision_recall_curve(
#     training_labels_5, decision_scores, pos_label=1
# )
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# thresh = -20000
# prediction = np.zeros(decision_scores.shape)
# prediction[decision_scores > thresh] = 1

# precision = precision_score(training_labels_5, prediction)
# recall = recall_score(training_labels_5, prediction)

# # Relevanz, Precision: TP / (TP + FP)

# # True Positive Rate = Recall (hit / hit + omission)
# # TPR = TP / (TP + FN)

# # False Positive Rate (false alarm / false alarm + correct rejection)
# # FPR = FP / (FP + TN)

# # Receiver Operator Characteristic = True Positive Rate auf Y / False Positive Rate auf X
# fpr, tpr, thresholds = roc_curve(training_labels_5, decision_scores)

# # Area under curve ROC_AUC
# auc_score = roc_auc_score(training_labels_5, decision_scores)

# # Interpret ROC: Diagonalen = Zufall,
# def plot_roc_curve(fpr, tpr, title=None):
#     plt.title(title)
#     plt.plot(fpr, tpr, linewidth=2)
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")


# plot_roc_curve(fpr, tpr, title=f"ROC AUC = {auc_score:.4f}")
# plt.show()

# Pick an example image
# one_image = test_data[2]
# one_image_2d = one_image.reshape(28, 28)

# Plot example image
# matplotlib.pyplot.imshow(
#     one_image_2d, cmap=matplotlib.cm.binary, interpolation="nearest"
# )
# matplotlib.pyplot.axis("off")
# matplotlib.pyplot.show()
