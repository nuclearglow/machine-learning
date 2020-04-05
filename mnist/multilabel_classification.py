import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

training_data = joblib.load("data/mnist_training_data.pkl")
training_labels = joblib.load("data/mnist_training_labels.pkl")

test_data = joblib.load("data/mnist_test_data.pkl")
test_labels = joblib.load("data/mnist_test_labels.pkl")

training_labels = np.array(training_labels, dtype=np.float64)

training_labels_large = training_labels >= 7
training_labels_odd = training_labels % 2 == 1
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
training_labels_multi = np.c_[training_labels_large, training_labels_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(training_data, training_labels_multi)

# directly predict training data index for multillabels and display along with training data label
idx = 5
prediction = np.c_[knn_clf.predict([training_data[idx, :]]), training_labels[idx]]


# Compute F1 scores for all labels and average across labels. An averaged F1 score so to say...
knn_prediction = cross_val_predict(knn_clf, training_data, training_labels_multi, cv=3)
knn_f1_score = f1_score(
    training_labels_multi, knn_prediction, average="macro"
)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
