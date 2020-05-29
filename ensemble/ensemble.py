#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Make some mooooons
X, y = make_moons(n_samples=10000, noise=0.25)

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize 3 different classifiers
log_clf = LogisticRegression()
forest_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

# Initialize a voting clssifier
voting_clf = VotingClassifier(
    estimators=[("log_clf", log_clf), ("forest", forest_clf), ("svc", svm_clf)],
    voting="soft",
)

# Train each classifier on data and print accuracy
for clf in (log_clf, forest_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)

    score = accuracy_score(y_test, y_prediction)
    clf_name = clf.__class__.__name__
    print(f"{clf_name} Accuracy Score: {score}")
