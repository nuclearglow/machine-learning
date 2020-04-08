from sklearn.base import BaseEstimator, TransformerMixin

# A sklearn transformer to work with DataFrames: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# Pandas Source: https://github.com/pandas-dev/pandas/releases


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data[self.attribute_names].to_numpy()
