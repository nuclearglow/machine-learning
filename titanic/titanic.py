#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import math
import joblib

import matplotlib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from transformers.DataFrameSelector import DataFrameSelector
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

from util import evaluate_classifier_model

## load data
train_csv = os.path.join("data", "train.csv")
training_data = pd.read_csv(train_csv)

test_csv = os.path.join("data", "test.csv")
test_data = pd.read_csv(test_csv)

# preprocessing

# check for missing data in label -> none
# training_data["Survived"].isna()
# invalid_training_labels = training_data[training_data["Survived"].isna()]

numeric_attributes = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
category_attributes = ["Pclass", "Embarked"]
binary_attributes = ["Sex", "Cabin"]

# TODO: move to pipeline, own transformer DataMapper
# prepare Sex category
training_data["Sex"] = training_data["Sex"].map({"male": 0, "female": 1})
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
# prepare Cabin data (String or NaN)
training_data["Cabin"] = training_data["Cabin"].map(
    lambda cabin: 1 if isinstance(cabin, str) else 0
)
test_data["Cabin"] = test_data["Cabin"].map(
    lambda cabin: 1 if isinstance(cabin, str) else 0
)

category_binarize_pipeline = Pipeline(
    [("selector", DataFrameSelector(binary_attributes)),]
)

category_onehot_pipeline = Pipeline(
    [
        ("selector", DataFrameSelector(category_attributes)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encode", OneHotEncoder(sparse=False)),
    ]
)

numeric_pipeline = Pipeline(
    [
        ("selector", DataFrameSelector(numeric_attributes)),
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)

preprocessing_pipeline = FeatureUnion(
    transformer_list=[
        ("numeric_pipeline", numeric_pipeline),
        ("binarize", category_binarize_pipeline),
        ("one_hot_encode", category_onehot_pipeline),
    ]
)

# extract labels
training_labels = training_data["Survived"].to_numpy()
# test_labels = test_data["Survived"].to_numpy()

# transform the date
titanic_training_data_preprocessed = preprocessing_pipeline.fit_transform(training_data)
# titanic_test_data_preprocessed = preprocessing_pipeline.fit_transform(test_data)

# joblib dump
joblib.dump(
    titanic_training_data_preprocessed, "data/titanic_training_data_preprocessed.pkl"
)
joblib.dump(training_labels, "data/titanic_training_labels.pkl")
# joblib.dump(titanic_test_data_preprocessed, "data/titanic_test_data_preprocessed.pkl")
# joblib.dump(test_labels, "data/titanic_test_labels.pkl")

# SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_validate = cross_validate(
    sgd_clf,
    titanic_training_data_preprocessed,
    training_labels,
    cv=3,
    scoring="accuracy",
)

sgd_cross_validation = cross_val_score(
    sgd_clf,
    titanic_training_data_preprocessed,
    training_labels,
    cv=3,
    scoring="accuracy",
)

decision_scores = cross_val_predict(
    sgd_clf,
    titanic_training_data_preprocessed,
    training_labels,
    cv=3,
    method="decision_function",
)

sgd_evaluation = evaluate_classifier_model(decision_scores, training_labels)

# Random Forest

random_forest_clf = RandomForestClassifier(random_state=42)

sgd_cross_validation = cross_val_score(
    random_forest_clf,
    titanic_training_data_preprocessed,
    training_labels,
    cv=3,
    scoring="accuracy",
)

decision_scores = cross_val_predict(
    random_forest_clf, titanic_training_data_preprocessed, training_labels, cv=3
)

random_forest_evaluation = evaluate_classifier_model(decision_scores, training_labels)
