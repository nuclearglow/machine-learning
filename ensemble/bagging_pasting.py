#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

# Make some mooooons
X, y = make_moons(n_samples=10000, noise=0.25)

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=False,  # True = bagging, False = pasting
    n_jobs=multiprocessing.cpu_count() - 2,
)

bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
bag_score = accuracy_score(y_test, y_pred_bag)


# comparison with single dec tree
dec_clf = DecisionTreeClassifier()
dec_clf.fit(X_train, y_train)
y_pred_dec = dec_clf.predict(X_test)
dec_score = accuracy_score(y_test, y_pred_dec)

# PLot
fig = plt.figure()
ax = plt.subplot(1, 2, 1)
fig = plot_decision_regions(X_train, y_train, clf=dec_clf, legend=2)
ax = plt.subplot(1, 2, 2)
fig = plot_decision_regions(X_train, y_train, clf=dec_clf, legend=2)
plt.show()
