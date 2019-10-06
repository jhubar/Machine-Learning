"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from data import make_data1, make_data2
from plot import plot_boundary

x,y = make_data1(2000)
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x,y, test_size=150/2000, random_state=None)

def knn_plot(n_neighbors_input = 1, fname=""):
    model = KNeighborsClassifier(n_neighbors = n_neighbors_input)
    model.fit(x_data_train, y_data_train)
    y_pred = model.predict(x_data_test)
    plot_boundary(fname, model, x, y)
    return accuracy_score(y_data_test, y_pred)

def knn(n_neighbors_input = 1, fname=""):
    model = KNeighborsClassifier(n_neighbors = n_neighbors_input)
    model.fit(x_data_train, y_data_train)
    y_pred = model.predict(x_data_test)
    return accuracy_score(y_data_test, y_pred)

def knn_cv(n_neighbors_input = 1, fname=""):
    model = KNeighborsClassifier(n_neighbors = n_neighbors_input)
    return cross_val_score(model, x, y, cv = 10)


if __name__ == "__main__":
    iteration = 20
    n_neighbors = [1,5,10]
    goal = []

    for i in n_neighbors:
        goal = knn_plot(n_neighbors_input = i, fname = "n_neighbors_"+str(i))
        print("Neighbours ",str(i),": goal ",str(goal))

        goal = []
        for j in range(iteration):
            goal.append(knn(n_neighbors_input = i))

        print("Neighbours ",str(i),": mean ",str(np.mean(goal)),", standard deviation: ", str(np.std(goal)))

        goal = knn_cv(n_neighbors_input = i, fname = "n_neighbors_"+str(i))
        print("10-fold cross validation ",str(i),": score ",str(goal.mean()))
