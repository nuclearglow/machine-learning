import cProfile
import joblib
import numpy as np
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

training_data = joblib.load("data/mnist_training_data.pkl")
training_labels = joblib.load("data/mnist_training_labels.pkl")

test_data = joblib.load("data/mnist_test_data.pkl")
test_labels = joblib.load("data/mnist_test_labels.pkl")

cp = cProfile.Profile()
cp.enable()

training_labels = np.array(training_labels, dtype=np.float64)

training_labels_large = training_labels >= 7
training_labels_odd = training_labels % 2 == 1
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
training_labels_multi = np.c_[training_labels_large, training_labels_odd]

# Übungsaufgabe Page 105
# Use Grid Search to find best model for different hyperparameters

knn_clf = KNeighborsClassifier()
# knn_clf.fit(training_data, training_labels_multi)

parameter_grid = [{"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}]
grid_search = GridSearchCV(
    knn_clf, parameter_grid, cv=3, n_jobs=multiprocessing.cpu_count() - 1, verbose=100
)
grid_search.fit(training_data, training_labels)

cp.disable()
cp.print_stats()

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# directly predict training data index for multillabels and display along with training data label
# idx = 5
# prediction = np.c_[knn_clf.predict([training_data[idx, :]]), training_labels[idx]]
