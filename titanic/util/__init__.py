import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score


def intersection(list1, list2):
    return list(set(list1) & set(list2))


def difference(list1, list2):
    return set(list1) - set(list2)


# Interpret ROC: Diagonalen = Zufall,
def plot_roc_curve(fpr, tpr, title=None):
    plt.title(title)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


def evaluate_classifier_model(decision_scores, training_labels, threshold=0):
    # normalize by threshold
    prediction = np.zeros(decision_scores.shape)
    prediction[decision_scores > threshold] = 1

    # confusion matrix
    conf_mat = confusion_matrix(
        training_labels, prediction, labels=np.unique(training_labels)
    )

    # Relevanz, Precision: TP / (TP + FP)
    precision = precision_score(training_labels, prediction)

    # True Positive Rate = Recall (hit / hit + omission)
    # TPR = TP / (TP + FN)
    recall = recall_score(training_labels, prediction)

    # False Positive Rate (false alarm / false alarm + correct rejection)
    # FPR = FP / (FP + TN)
    # Receiver Operator Characteristic = True Positive Rate auf Y / False Positive Rate auf X
    fpr, tpr, thresholds = roc_curve(training_labels, decision_scores)

    # Area under curve ROC_AUC
    auc_score = roc_auc_score(training_labels, decision_scores)

    plot_roc_curve(fpr, tpr, title=f"ROC AUC = {auc_score:.4f}")
    plt.show()

    return dict(
        precision=precision,
        recall=recall,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc_score=auc_score,
        conf_ma=conf_mat,
    )
