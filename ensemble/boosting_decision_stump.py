#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Train random forest on iris
iris = load_iris()

# splitting
X, y = iris["data"], iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# adaboost classifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.5,
)

# cross validation
cross_validation_score = cross_val_score(
    ada_clf, X_train, y_train, cv=10, scoring="accuracy"
)
