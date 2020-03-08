from sklearn.base import BaseEstimator, TransformerMixin

# A sklearn transformer to select attributes from the dataset identified by parameter attribute_names (a list)
class AttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data.get(self.attribute_names)
