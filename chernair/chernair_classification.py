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
import joblib

# current directory with data file
preprocessed_data_path = os.path.abspath(
    f"{os.getcwd()}/data/chernair-preprocessing.pkl"
)

# Read data
data = joblib.load(preprocessed_data_path)

# 1 Klassifikator-Modell
#   cherntime, isotop-kontzentration ->Schätzt, welche Stadt
