from sklearn.base import BaseEstimator, TransformerMixin

# Takes a Pandas DataFrame and returns a numpy array
class DataFrameToNumpyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data.to_numpy()
