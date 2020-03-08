#!/usr/bin/env python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from transformers.DataFrameToNumpyTransformer import DataFrameToNumpyTransformer
from transformers.AttributeSelector import AttributeSelector
from transformers.LabelSelector import LabelSelector
from sklearn.pipeline import Pipeline, FeatureUnion
import joblib

# select the relevant numeric attributes
numeric_attributes = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
categorial_attributes = ["ocean_proximity"]
label = ["median_house_value"]

# Define our pipeline
# numeric part
numeric_pipeline = Pipeline(
    [
        ("select_numeric_attributes", AttributeSelector(numeric_attributes)),
        ("df_to_numpy", DataFrameToNumpyTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)

# extract label pipeline
label_pipeline = Pipeline(
    [
        ("get_labels", LabelSelector(label)),
        ("df_to_numpy", DataFrameToNumpyTransformer()),
    ]
)

# categorial part
category_pipeline = Pipeline(
    [
        ("select_categorial_attributes", AttributeSelector(categorial_attributes)),
        ("df_to_numpy", DataFrameToNumpyTransformer()),
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
        # ("label_binarizer", LabelBinarizer()),
        ("one_hot", OneHotEncoder(sparse=False)),
        # ("cat_encoder", CategorialEncoder(encoding="onehot-dense")),
    ]
)

# # combined parallel preprocessing pipeline
preprocessing_pipeline = FeatureUnion(
    transformer_list=[
        ("numeric_pipeline", numeric_pipeline),
        ("category_pipeline", category_pipeline),
    ]
)

# test the preprocessing pipeline TODO: remove?

# load raw data
training_data_raw = joblib.load("data/01_training_data_raw.pkl")
test_data_raw = joblib.load("data/01_test_data_raw.pkl")

# Run preprocessing on datasets
training_data = preprocessing_pipeline.fit_transform(training_data_raw)
test_data = preprocessing_pipeline.fit_transform(test_data_raw)

# access pipeline steps
# step = numeric_pipeline.named_steps["select_numeric_attributes"]

# Get labels
training_data_labels = label_pipeline.fit_transform(training_data_raw)
test_data_labels = label_pipeline.fit_transform(test_data_raw)

# TODO
# model_training_pipeline = Pipeline([])
