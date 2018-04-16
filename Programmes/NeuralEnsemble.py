# Import modules

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import csv
import time

from sklearn import decomposition
from sklearn import datasets

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier as GPC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import random

from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error

from statistics import mode


class NeuralEnsemble:
    """Algorithm that uses Gaussian Process Classifier and Support Vector
    Machine to imitate the Brain's architecture to solve problems"""

    def __init__(self, X, y_tar):
        """Data"""
        self.X = X
        self.y_tar = y_tar
        self.features_accuracy = []   # accuracy per feature
        self.features_index = []   # features with good accuracy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_tar, test_size=.2, random_state=42)
        self.counter = 0
        self.debug = []
        self.accuracy_list = []
        self.neuron_list = [136, 167, 140, 160, 147, 174, 150, 168, 168]

    def compute_per_gaussian(self, max_iter=100):
        """Compute SVM per feature"""

        print(len(self.X_train))
        print(len(self.X_train[0]))

        # per feature
        for feature_index in range(int(len(self.X[0]))):
            X_train_mod = []
            # define training dataset
            for example in range(len(self.X_train)):   # for each example (469)
                X_train_mod.append([self.X_train[example][self.counter]])

            X_test_mod = []
            # define testing dataset
            for example in range(len(self.X_test)):   # for each example (469)
                X_test_mod.append([self.X_test[example][self.counter]])

            clf = GPC(max_iter_predict=max_iter)  # GPC model
            clf.fit(X_train_mod, self.y_train) # compute with only one feature
            score = clf.score(X_test_mod, self.y_test)

            self.features_accuracy.append(score)

            self.counter += 1

    def define_indices(self, threshold=0.55, inverse=False):
        """Return indices above cerain accuracy threshold."""

        counter = 0  # used for counting the index of acuracies
        self.features_index = []
        if inverse:
            for item in self.features_accuracy:
                if item <= threshold:
                    self.features_index.append(counter)
                counter += 1
        else:
            for item in self.features_accuracy:
                if item > threshold:
                    self.features_index.append(counter)
                counter += 1


    def reuturn_data(self):
        pass

    def compute_reduced_svm(self, gamma_par='auto', C_par=1, kernel_par='rbf'):
        """SVM on select number of features"""

        X_train_mod = []
        # define training dataset
        for example in range(len(self.X_train)):   # for each example (469)
            temp = []
            for index in self.features_index:
                temp.append(self.X_train[example][index])
            X_train_mod.append(temp)

        self.debug = X_train_mod

        X_test_mod = []
        # define testing dataset
        for example in range(len(self.X_test)):   # for each example (469)
            temp = []
            for index in self.features_index:
                temp.append(self.X_test[example][index])
            X_test_mod.append(temp)


        # c, gamma, kernel = self.parameter_search_svm(self.X_train, self.y_train)

        clf = SVC(kernel=kernel_par, C=C_par, gamma=gamma_par)  # SVC model

        clf.fit(X_train_mod, self.y_train) # compute with only one feature
        score = clf.score(X_test_mod, self.y_test)
        return(score)

    def compute_reduced_mlp(self):
        """SVM on select number of features"""

        X_train_mod = []



        # define training dataset
        for example in range(len(self.X_train)):   # for each example (469)
            temp = []
            for index in self.features_index:
                temp.append(self.X_train[example][index])
            X_train_mod.append(temp)

        self.debug = X_train_mod

        X_test_mod = []
        # define testing dataset
        for example in range(len(self.X_test)):   # for each example (469)
            temp = []
            for index in self.features_index:
                temp.append(self.X_test[example][index])
            X_test_mod.append(temp)



        mlp = MLPClassifier(hidden_layer_sizes=(3,2), max_iter=3500, activation="tanh")  # Fit MLP() model
        mlp.fit(X_train_mod, self.y_train) # compute with only one feature
        score = mlp.score(X_test_mod, self.y_test)
        return(score)

            # Parameter search for SVM

    def parameter_search_svm(self, X_train_data, y_train_data, C=[0.1,10], gamma=[1e-5,1e-1], kernel=['linear', 'rbf'], verbose=True):

        """Parameter search for SVM. Returns best parameters in form: C, gamma, kernel

        X_train_data: X

        y_train_data: y

        Default parameters are arguments"""

        parameters = [{'C': C, 'gamma': gamma, 'kernel': kernel}]

        svc = SVC()

        clf = GridSearchCV(svc, parameters)

        clf.fit(X_train_data, y_train_data)     # Fit model

        if verbose:   # If verbose, export parameters

            means = clf.cv_results_['mean_test_score']

            params = clf.cv_results_['params']

            for mean, param in zip(means, params):

                print("{}\t{}".format(round(mean, 3), param))



        # Return best parameters in list form
        print("param")
        return clf.best_estimator_.C, clf.best_estimator_.gamma, clf.best_estimator_.kernel

    def update_epoch(self):
        self.epoch += 1

    def return_indices(self):
        return self.features_index
