#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: plkn, nuky
"""

import tensorflow as tf
import pandas as pd

# datasets
x = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset_same = tf.data.Dataset.range(10)

for item in dataset:
    print(item)

# Dataset Transformation methods

dataset1 = dataset.repeat(3)
for item in dataset1:
    print(item)

dataset2 = dataset.batch(7)
for item in dataset2:
    print(item)

dataset3 = dataset.batch(7, drop_remainder=True)
for item in dataset3:
    print(item)


dataset4 = dataset.batch(3).unbatch()
for item in dataset4:
    print(item)

# Filter dataset
dataset5 = dataset.filter(lambda x: x < 5)
for item in dataset5:
    print(item)

# data item transformation
dataset6 = dataset.map(lambda i: i ** 2)

for item in dataset6:
    print(item)

# Shuffle dataset6
dataset7 = tf.data.Dataset.range(30).shuffle(buffer_size=5).batch(5)
for item in dataset7:
    print(item)


HOUSING_PATH = "datasets/housing"

# Load housing csv file
df_housing = pd.read_csv("data/housing.csv", header=0)

# Pop y
y = df_housing.pop("median_house_value")

# Ocean as categories
df_housing["ocean_proximity"] = df_housing["ocean_proximity"].astype("category")
df_housing["ocean_proximity_cat"] = pd.to_numeric(
    df_housing["ocean_proximity"].cat.codes
)

# Pop ocean_proximity strings
op = df_housing.pop("ocean_proximity")

# Convert to tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((df_housing.values, y.values))

