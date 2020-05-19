#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
import numpy as np

# Load petal data of iris dataset
iris = load_iris()
x = iris.data[:, 2:]  # Petal length and wid
# x = x[:, [1, 0]]
x = np.flip(x, 1)
y = iris.target

# Initialize and fit a decision Tree clssifier
depth = None
tree_clf = DecisionTreeClassifier(max_depth=depth, criterion="entropy")
tree_clf.fit(x, y)

out = f"iris_tree_depth_{depth}_entropy"

# create a classifier representation
export_graphviz(
    tree_clf,
    out_file=f"{out}.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True,
)

# Install Prerequisites
# sudo apt-get install graphviz
# graphviz docs: https://graphviz.gitlab.io/documentation/
#
# convert dot file to png like tbhis
# dot -Tpng iris_tree.dot -o iris_tree.png
subprocess.run(["dot", "-Tpng", f"{out}.dot", f"-o {out}.png"])

# decision trees can predict probabilities
prediction = tree_clf.predict_proba(x)
