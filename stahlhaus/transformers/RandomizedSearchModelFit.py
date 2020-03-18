from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV

# A sklearn transformer to fit data to find the best estimator for a given model using randomized cross validation search
class RandomizedSearchModelFit(BaseEstimator, TransformerMixin):
    def __init__(self, model, param_distributions, scoring=None, cv=None, n_iter=10):
        self.model = model
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter

    def fit(self, data, labels):
        self.result = RandomizedSearchCV(
            self.model,
            self.param_distributions,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
        )
        self.result.fit(data, labels)

    def transform(self, data):
        return self.result.best_estimator_
