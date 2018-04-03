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


class BrainAlgorithm:
    def __init__(self, X, y_tar, epoch, session=0):
        """Data"""
        self.X = X
        self.y_tar = y_tar
        self.features_accuracy = []   # accuracy per feature
        self.features_index = []   # features with good accuracy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_tar, test_size=.2, random_state=42)
        self.counter=0
        self.debug = []
        self.epoch = epoch
        self.accuracy_list = []
        self.session = session
        self.neuron_list = [136, 167, 140, 160, 147, 174, 150, 168, 168]
        self.neuron_num = self.neuron_list[self.session]

    def compute_per_svm(self):
        """Compute SVM per feature"""
        # per feature
        for feature_index in range(int(len(X[0])/45)):
            X_train_mod = []
            # define training dataset
            for example in range(len(self.X_train)):   # for each example (469)
                X_train_mod.append([self.X_train[example][self.epoch*self.neuron_num + self.counter]])

            X_test_mod = []
            # define testing dataset
            for example in range(len(self.X_test)):   # for each example (469)
                X_test_mod.append([self.X_test[example][self.epoch*self.neuron_num + self.counter]])

            gamma = 1e-2
            c = 10
            kernel = 'linear'
            # c, gamma, kernel = self.parameter_search_svm(self.X_train, self.y_train)

            clf = SVC(kernel=kernel, C=c, gamma=gamma)  # SVC model
            clf.fit(X_train_mod, self.y_train) # compute with only one feature
            score = clf.score(X_test_mod, self.y_test)  # score of individual neuron on ~400 tasks per defined epoch
            self.features_accuracy.append(score)  # accuracy of neuron on that example on ~400 tasks per defined epoch

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

    def compute_reduced_svm(self):
        """SVM on select number of features"""

        X_train_mod = []
        # define training dataset
        for example in range(len(self.X_train)):   # for each example (469)
            temp = []
            for index in self.features_index:
                temp.append(self.X_train[example][self.epoch*self.neuron_num + index])
            X_train_mod.append(temp)

        self.debug = X_train_mod

        X_test_mod = []
        # define testing dataset
        for example in range(len(self.X_test)):   # for each example (469)
            temp = []
            for index in self.features_index:
                temp.append(self.X_test[example][self.epoch*self.neuron_num + index])
            X_test_mod.append(temp)

        gamma = 1e-1
        c = 10
        kernel = 'poly'
        # c, gamma, kernel = self.parameter_search_svm(self.X_train, self.y_train)

        clf = SVC(kernel=kernel, C=c, gamma=gamma)  # SVC model

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
                temp.append(self.X_train[example][15*136 + index])
            X_train_mod.append(temp)

        self.debug = X_train_mod

        X_test_mod = []
        # define testing dataset
        for example in range(len(self.X_test)):   # for each example (469)
            temp = []
            for index in self.features_index:
                temp.append(self.X_test[example][15*136 + index])
            X_test_mod.append(temp)

        gamma = 1e-1
        c = 0.1
        kernel = 'poly'
        # c, gamma, kernel = self.parameter_search_svm(self.X_train, self.y_train)

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

    def update_epoch():
        self.epoch += 1

        class VotingBrainAlgorithm(BrainAlgorithm):

            def __init__(self, X, y_tar, epoch, session=0):
                """Data"""
                self.X = X
                self.y_tar = y_tar
                self.features_accuracy = []   # accuracy per feature
                self.features_index = []   # features with good accuracy
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_tar, test_size=.2, random_state=42)
                self.counter=0
                self.debug = []
                self.epoch = epoch
                self.accuracy_list = []
                self.session = session
                self.neuron_list = [136, 167, 140, 160, 147, 174, 150, 168, 168]
                self.neuron_num = self.neuron_list[self.session]

            def compute_per_gaussian(self, max_iter=100):
                """Compute SVM per feature"""
                # per feature
                for feature_index in range(int(len(X[0])/45)):
                    X_train_mod = []
                    # define training dataset
                    for example in range(len(self.X_train)):   # for each example (469)
                        X_train_mod.append([self.X_train[example][self.epoch*self.neuron_num + self.counter]])

                    X_test_mod = []
                    # define testing dataset
                    for example in range(len(self.X_test)):   # for each example (469)
                        X_test_mod.append([self.X_test[example][self.epoch*self.neuron_num + self.counter]])

                    gamma = 1e-2
                    c = 10
                    kernel = 'linear'

                    clf = GPC(max_iter_predict=max_iter)  # GPC model
                    clf.fit(X_train_mod, self.y_train) # compute with only one feature
                    score = clf.score(X_test_mod, self.y_test)

                    self.features_accuracy.append(score)

                    self.counter += 1

            def compute_reduced_knn(self, n_neigh=3):
                """Compute with KNN"""
                X_train_mod = []
                # define training dataset
                for example in range(len(self.X_train)):   # for each example (469)
                    temp = []
                    for index in self.features_index:
                        temp.append(self.X_train[example][self.epoch*self.neuron_num + index])
                    X_train_mod.append(temp)

                self.debug = X_train_mod

                X_test_mod = []
                # define testing dataset
                for example in range(len(self.X_test)):   # for each example (469)
                    temp = []
                    for index in self.features_index:
                        temp.append(self.X_test[example][self.epoch*self.neuron_num + index])
                    X_test_mod.append(temp)

                gamma = 1e-1
                c = 10
                kernel = 'poly'
                # c, gamma, kernel = self.parameter_search_svm(self.X_train, self.y_train)

                neigh = KNeighborsClassifier(n_neighbors=n_neigh)  # SVC model

                neigh.fit(X_train_mod, self.y_train) # compute with only one feature
                score = neigh.score(X_test_mod, self.y_test)
                return(score)


            def voting_svm(self):
                """Voting implementation of SVM for a unique epoch"""
                per_neuron_prediction = []

                """
                STRUCTURE:
                -> Key neurons
                    -> Each epoch
                        -> Number of tasks (~100)
                            -> Results for each neuron
                """

                # Choosing features
                # train data
                print("test")
                for neuron in self.features_index:    # for good neurons
                    neuron_votes = []
                    X_for_neuron = []
                    for example in range(len(self.X_train)):     # for each of tasks
                        X_for_neuron.append([self.X_train[example][self.epoch*self.neuron_num + neuron]])

                    X_test = []
                    for example in range(len(self.X_test)):     # for each of tasks
                        X_test.append([self.X_test[example][self.epoch*self.neuron_num + neuron]])

                    clf = GPC()
                    # prediction on individual neuron
                    clf.fit(X_for_neuron, self.y_train)
                    # add predictions to data for each sample
                    pred = clf.predict(X_test)


                    neuron_votes.append(pred)
                    per_neuron_prediction.append(neuron_votes)


                # test data
                accuracy = 0
                print(per_neuron_prediction[0])
                print(len(self.X_test))

                features_num = len(self.features_index)

                # check if voting legnth is even
                """
                if len(per_neuron_prediction)%2==0:
                    del per_neuron_prediction[-1]
                    features_num =- 1
                """

                print(per_neuron_prediction)

                # for each testing task per session per epoch
                for test_task in range(len(self.X_test)):
                    # count the most number of votes as predicted by SVC
                    # classifier per individual neuron
                    temp_task = []
                    for neuron in range(features_num):
                        temp_task.append(per_neuron_prediction[neuron][0][test_task])
                    vote_result = mode(temp_task)
                    if vote_result == self.y_test[test_task]:
                        accuracy += 1
                    print("ACCURACY {}".format(accuracy/(test_task+1)))

                accuracy = accuracy/len(self.X_test)

                return accuracy

                # return accuracy


            def compute_reduced_svm(self):
                """SVM on select number of features"""

                X_train_mod = []
                # define training dataset
                for example in range(len(self.X_train)):   # for each example (469)
                    temp = []
                    for index in self.features_index:
                        temp.append(self.X_train[example][self.epoch*self.neuron_num + index])
                    X_train_mod.append(temp)

                self.debug = X_train_mod

                X_test_mod = []
                # define testing dataset
                for example in range(len(self.X_test)):   # for each example (469)
                    temp = []
                    for index in self.features_index:
                        temp.append(self.X_test[example][self.epoch*self.neuron_num + index])
                    X_test_mod.append(temp)

                gamma = 1e-1
                c = 10
                kernel = 'poly'
                # c, gamma, kernel = self.parameter_search_svm(self.X_train, self.y_train)

                clf = SVC(kernel=kernel, C=c, gamma=gamma)  # SVC model

                clf.fit(X_train_mod, self.y_train) # compute with only one feature
                score = clf.score(X_test_mod, self.y_test)
                return(score)
