#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  2_visualizing.py
#  
#  Copyright 2016 Ajay Bhatia <prof.ajaybhatia@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  Visualizing a Decision Tree
# 
#  Goals:
#	 1. Import Iris dataset.
#	 2. Train a classifier.
#	 3. Predict label for new flower.	
#	 4. Visualizes the tree.

import numpy as np
import pydotplus

from sklearn.datasets import load_iris
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn import tree

# Load Iris dataset
iris = load_iris()
'''
# Print features metadata
print(iris.feature_names)
# Print targets "labels" metadata
print(iris.target_names)
# Print first row of features(data)
print(iris.data[0])
# Print first row of target "label"(data)
print(iris.target[0])
'''
test_idx = [0, 50, 100]

# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Decision Tree classifier
clf = tree.DecisionTreeClassifier()
# Training algorithm
clf = clf.fit(train_data, train_target)

# See what's actually in test data
print(test_target)
# See what our classifier predicts
print(clf.predict(test_data))

# Visualize
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
						feature_names=iris.feature_names,
						class_names=iris.target_names,
						filled=True, rounded=True,
						special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")