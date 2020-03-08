from sklearn.base import BaseEstimator, TransformerMixin

# A sklearn transformer to drop the label attribute from the dataset and return it
class LabelSelector(BaseEstimator, TransformerMixin):
    def __init__(self, label_name):
        self.label_name = label_name

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        return data.get(self.label_name)
