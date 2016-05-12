#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  1_hello_world.py
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
#  Six Lines of code it all needs to write 
#  your first machine learning program

from sklearn import tree

# 0 - stands for "bumpy" and 1 stands for "smooth"
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 stands for "apple" and 1 stands for "orange"
labels = [0, 0, 1, 1]

# Classifier - Decision Tree
clf = tree.DecisionTreeClassifier()
# Learning Algorithm
clf = clf.fit(features, labels) # fit = 'f'ind patterns 'i'n da't'a
# Predict a fruit
print(clf.predict([[150, 0]]))
