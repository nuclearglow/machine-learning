import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve

# Keras Tensorboard Callback
def get_run_logdir(logpath=os.path.join(os.curdir, "tensorboard_logs")):
    run_id = time.strftime("run-%Y-%m-%d-%H-%M-%S")
    return os.path.join(logpath, run_id)


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


def plot_learning_curve(
    estimator,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=multiprocessing.cpu_count() - 1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring=None,
    title="Learning Curve",
):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=multiprocessing.cpu_count() - 1)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    # learning curves in scikit learn
    # https://devdocs.io/scikit_learn/modules/generated/sklearn.model_selection.learning_curve#sklearn.model_selection.learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs, scoring=scoring
    )

    # https://devdocs.io/scikit_learn/auto_examples/model_selection/plot_learning_curve#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if ylim is not None:
        plt.ylim(*ylim)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="#00AAAA",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="#AA00AA",
    )
    plt.plot(
        train_sizes, train_scores_mean, "o-", color="#00AAAA", label="Training score"
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color="#AA00AA",
        label="Cross-validation score",
    )

    # plt.axis([0, 80, 0, 3])

    plt.legend(loc="best")
    plt.show()
