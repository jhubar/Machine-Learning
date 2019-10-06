"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary

X,y = make_data1(2000)
# data_train  = dataset[0:149]
# data_test = dataset[150:]

x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(X,y, test_size=150/2000, random_state=None)
# data_test, data_hold = train_test_split(data_test_hold, test_size=0.33, random_state=21)

def tree(max_depth_input = None, fname = ""):
    model = DecisionTreeClassifier(criterion='gini', splitter='best',
                     max_depth = max_depth_input, min_samples_split=2,
                     min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                     max_features=None, random_state=None,
                     max_leaf_nodes=None, min_impurity_decrease=0.0,
                     min_impurity_split=None, class_weight=None,
                     presort=False)

    model.fit(x_data_train, y_data_train)

    y_pred = model.predict(x_data_test)

    plot_boundary(fname, model, X, y)

    return accuracy_score(y_data_test, y_pred)

#print("Model Accuracy: %.2f" % (accuracy_score(y_data_test,y_pred)*100), "%")



if __name__ == "__main__":
    iteration = 20
    goal = []
    depths = [1,2,4,8, None]

    for i in depths :
        goal = tree(max_depth_input = i, fname = "depth_"+str(i))
        goal = []

        for j in range(iteration):
            goal.append(tree(max_depth_input = i))
        print("Depth ",str(i),": mean ",str(np.mean(goal)),", standard deviation: ",str(np.std(goal)))
