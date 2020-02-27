import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, data, y=None):
        return self

    # demo: pappe kram dran, abhängig vm paramster in init
    def transform(self, data, y=None):
        rooms_per_household = data[:, rooms_ix] / data[:, household_ix]
        population_per_household = data[:, population_ix] / data[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = data[:, bedrooms_ix] / data[:, rooms_ix]
            return np.c_[
                data, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[data, rooms_per_household, population_per_household]
