#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeRegressor, export_graphviz
import subprocess
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-0.5, 0.5001, 0.001).reshape(-1, 1)
y = x ** 2 + np.random.randn(x.shape[0], x.shape[1]) * 0

# plot data
plt.plot(x, y)

# Initialize and fit a decision Tree clssifier
depth = 2
tree_reg = DecisionTreeRegressor(max_depth=depth)
tree_reg.fit(x, y)

out = f"quadratic_noise_{depth}"

# create a classifier representation
export_graphviz(
    tree_reg, out_file=f"{out}.dot", rounded=True, filled=True,
)

# Install Prerequisites
# sudo apt-get install graphviz
# graphviz docs: https://graphviz.gitlab.io/documentation/
#
# convert dot file to png like tbhis
# dot -Tpng iris_tree.dot -o iris_tree.png
subprocess.run(["dot", "-Tpng", f"{out}.dot", f"-o {out}.png"])

# decision trees can predict probabilities
# prediction = tree_clf.predict_proba(x)

predict = tree_reg.predict([[84]])
