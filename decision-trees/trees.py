#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess


# Load petal data of iris dataset
iris = load_iris()
x = iris.data[:, 2:]  # Petal length and wid
y = iris.target

depth = 3
# Initialize and fit a decision Tree clssifier
tree_clf = DecisionTreeClassifier(max_depth=depth)
tree_clf.fit(x, y)

out = f"iris_tree_depth_{depth}"

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
