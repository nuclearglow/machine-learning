#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC


# Load MNIST dataset
mnist_X = joblib.load("data/mnist_training_data.pkl")
mnist_y = joblib.load("data/mnist_training_labels.pkl")

# Split dataset
X_train, X_splitted, y_train, y_splitted = train_test_split(
    mnist_X, mnist_y, test_size=20000, stratify=mnist_y
)
X_test, X_validate, y_test, y_validate = train_test_split(
    X_splitted, y_splitted, test_size=10000, stratify=y_splitted
)

# Random Forest
forest_clf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    max_depth=None,
    n_jobs=multiprocessing.cpu_count() - 2,
)
forest_clf.fit(X_train, y_train)

random_forest_prediction = forest_clf.predict(X_validate)
accuracy_random_forest = accuracy_score(
    y_validate, random_forest_prediction, normalize=True
)

extra_tree_clf = ExtraTreesClassifier(
    n_estimators=500,
    random_state=42,
    max_depth=None,
    n_jobs=multiprocessing.cpu_count() - 2,
)
extra_tree_clf.fit(X_train, y_train)

extra_tree_prediction = extra_tree_clf.predict(X_validate)
accuracy_extra_tree = accuracy_score(y_validate, extra_tree_prediction, normalize=True)


svm_clf = Pipeline(
    [("scaler", StandardScaler()), ("linear_svc", SVC(probability=True, kernel="rbf")),]
)
svm_clf.fit(X_train, y_train)
svm_prediction = svm_clf.predict(X_validate)
accuracy_svm = accuracy_score(y_validate, svm_prediction, normalize=True)

# Initialize a voting clssifier
voting_clf = VotingClassifier(
    estimators=[
        ("random_forest", forest_clf),
        ("extra_tree_clf", extra_tree_clf),
        ("svm", svm_clf),
    ],
    voting="soft",
)
voting_clf.fit(X_train, y_train)
voting_clf_prediction = voting_clf.predict(X_validate)
accuracy_voting = accuracy_score(y_validate, voting_clf_prediction, normalize=True)

