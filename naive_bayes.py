"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


from data import make_data1, make_data2
from plot import plot_boundary


class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):


    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    def fit(self, X, y):
        """Fit a Gaussian navie Bayes model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")


        return self._partial_fit(X, y, np.unique(y), _refit=True, sample_weight=sample_weight)




    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        return np.exp(self.predict_log_proba(X))

if __name__ == "__main__":
    from data import make_data1
    from plot import plot_boundary
