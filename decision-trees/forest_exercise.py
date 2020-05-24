#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import itertools

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Make moons
X, y = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# shuffle split
n_splits = 1000
shuffle_split = ShuffleSplit(n_splits=n_splits, train_size=100, random_state=42)

# Init Decision Tree
tree_clf = DecisionTreeClassifier(max_depth=7, criterion="entropy", max_leaf_nodes=None)

# Initialize accuracy score array
accuracy_scores = np.zeros((n_splits))

# Matrix for predicted values
y_predictions = np.zeros((n_splits, X_test.shape[0]))

for i, (sample_index, complement_index) in enumerate(shuffle_split.split(X_train)):
    tree_clf.fit(X_train[sample_index, :], y_train[sample_index])

    y_predictions[i, :] = tree_clf.predict(X_test)
    accuracy_scores[i] = accuracy_score(y_test, y_predictions[i, :], normalize=True)

majority_predictions, counts = mode(y_predictions, axis=0)


random_forest_accuracy_score = accuracy_score(
    y_test, majority_predictions.ravel(), normalize=True
)
