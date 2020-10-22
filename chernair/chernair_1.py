#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import numpy as np
import os
import geopy.distance
import datetime
import multiprocessing
import itertools
from pytz import timezone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path

# current directory with data file
data_path = os.path.abspath(f"{os.getcwd()}/CHERNAIR.csv")

# Read data
data = pd.read_csv(data_path)

# Rename columns
data.rename(
    columns={
        "PAYS": "country",
        "Ville": "city",
        "Y": "lat",
        "X": "lng",
        "I 131 (Bq/m3)": "I-131",
        "Cs 134 (Bq/m3)": "Cs-134",
        "Cs 137 (Bq/m3)": "Cs-137",
    },
    inplace=True,
)

# Chernobyl explosion coordinates
chern_lat_lng = (51.386998452, 30.092666296)

# Create chernobyl explosion time in UTC
cherntime_naive = datetime.datetime(1986, 4, 26, 1, 23, 40)
chern_tz = timezone("Europe/Kiev")
cherntime_aware = chern_tz.localize(cherntime_naive)

# Get distance from explosion in km
def get_distance_from_chernobyl_explosion(row):
    return geopy.distance.distance(tuple(row), chern_lat_lng).km


data["cherndist"] = data[["lat", "lng"]].apply(
    get_distance_from_chernobyl_explosion, axis=1
)

# Get time from explosion in seconds
def get_seconds_from_chernobyl_explosion(row):
    if row[1].count(":") == 1:
        row[1] = f"{row[1]}:00"
    year, month, day = [int(x) for x in row[0].split("/")]
    year += 1900
    hour, minute, second = [int(x) for x in row[1].split(":")]
    incday = False
    if hour == 24:
        hour = 0
        incday = True
    sampletime_naive = datetime.datetime(year, month, day, hour, minute, second)
    if incday:
        sampletime_naive = sampletime_naive + datetime.timedelta(days=1)
    sample_tz = timezone("CET")
    sampletime_aware = sample_tz.localize(sampletime_naive)
    return (sampletime_aware - cherntime_aware).total_seconds()


data["cherntime"] = data[["Date", "End of sampling"]].apply(
    get_seconds_from_chernobyl_explosion, axis=1
)

# Isotopes to numeric
data["I-131"] = pd.to_numeric(data["I-131"], errors="coerce")
data["Cs-134"] = pd.to_numeric(data["Cs-134"], errors="coerce")
data["Cs-137"] = pd.to_numeric(data["Cs-137"], errors="coerce")

# Country and city as categorial and add label
data["country"] = data["country"].astype("category")
data["city"] = data["city"].astype("category")
data["country_code"] = data["country"].cat.codes
data["city_code"] = data["city"].cat.codes


# interpolate missing values using regression
def interpolate_missing_isotope_values(data, isotope, predictors):

    # Add new column
    data[f"{isotope}_interpolated"] = data[isotope]

    # Exclude missing data
    training_data = data.dropna(subset=[isotope])
    X = training_data.filter(items=predictors).to_numpy()
    y = training_data.filter(items=[isotope]).to_numpy()

    # Get missing data predictors
    forest_reg = RandomForestRegressor(
        n_estimators=1000, n_jobs=multiprocessing.cpu_count() - 1
    )
    forest_reg.fit(X, y)
    missing_idx = data[isotope].isna()
    prediction_data = data.filter(items=predictors).copy()[missing_idx].to_numpy()

    # Predict missin values
    data[f"{isotope}_interpolated"][missing_idx] = forest_reg.predict(
        prediction_data
    ).ravel()
    return data


# one-hot encode cities
interpolated_data = data.copy(deep=True)
one_hot_encoder = OneHotEncoder(categories="auto")
df_expansion = pd.DataFrame(
    one_hot_encoder.fit_transform(interpolated_data[["city_code"]]).toarray()
)
interpolated_data = pd.concat([interpolated_data, df_expansion], axis=1, sort=False)

predictors = ["cherntime"]
one_hot_predictors = list(range(0, 95))
predictors = predictors + one_hot_predictors

for isotope in ["I-131", "Cs-134", "Cs-137"]:
    interpolated_data = interpolate_missing_isotope_values(
        interpolated_data, isotope, predictors
    )

interpolated_data.drop(labels=one_hot_predictors, axis=1, inplace=True)
