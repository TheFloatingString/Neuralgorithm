"""
Written by Laurence Liang
Updated as of December 30th, 2017
"""

# Import modules

# Read and initialize data
import csv
import pandas as pd
import numpy as np

# KNN
from sklearn import neighbors, datasets

# SVMx
from sklearn.svm import SVC, LinearSVC

# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# MLP
from sklearn.neural_network import MLPClassifier

# Train test split
from sklearn.model_selection import train_test_split

# Parameter search
from sklearn.model_selection import GridSearchCV

# Data scaling
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

# Data normalization
from sklearn.preprocessing import  Normalizer, normalize

# Matplotlib
import matplotlib.pyplot as plt

# Time
import time


class NeuronDecoding:

    def __init__(self, X=[], y_cue=[], y_loc=[], y_tar=[], neurons_per_session=[146, 133, 148, 146, 129, 132, 136, 141, 131], number_of_epochs=45):
        """Initialize instances"""
        self.X = []         # float[][]     # Data
        self.y_cue = []     # float[]
        self.y_loc = []     # float[]
        self.y_tar = []     # float[]
        self.neurons_per_session = [136, 167, 140, 160, 147, 174, 150, 168, 168]
        self.number_of_epochs = 45      # Number of epochs

    def read_global_data(self, number_of_sessions=9,
                         list_of_main_tasks=['K040217', 'K050217', 'K070217', 'K200117', 'K220117', 'K230117',
                                             'K260117', 'K300117', 'K310117']):
        """Read data from source"""
        # Source pathname
        # All predictor (X) data and all label (y) data
        # Add each session as a list in self.X and self.y_cue, self.y_loc, self.y_tar
        for session in range(number_of_sessions):
            # Temporary container variable
            X_temporary = []
            y_cue_temporary = []
            y_loc_temporary = []
            y_tar_temporary = []

            # Read data from corresponding folder

            # Open predictor file
            with open(str('Laurence_Data/Main_Task/' + list_of_main_tasks[session] + '/Data_matrix.csv'),'r') as csvfile:
                # Read predictor file
                reader = csv.reader(csvfile, delimiter=',') # Read CSV file using ',' as delimiter

                for row in reader:
                    int_row = list(map(float, row)) # Map String to float per 1D array
                    X_temporary.append(int_row)     # Store data from predictor file as 2D array

            try:
                # Read cue label
                with open('Laurence_Data/Main_Task/'+ list_of_main_tasks[session] + '/Labels_cue.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    y = []
                    for row in reader:
                        int_row = list(map(float, row)) # Map String to float per 1D array
                        y.append(int_row)               # Store data from label file as 2D array

                # Flatten list
                for i in y[0]:
                    y_cue_temporary.append(i)                        # Flatten   2D array to 1D array
            except:
                print("No cue label.")

            # Read location label
            with open('Laurence_Data/Main_Task/' + list_of_main_tasks[session] + '/Lbales_location.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                y = []
                for row in reader:
                    int_row = list(map(float, row))  # Map String to float per 1D array
                    y.append(int_row)                # Store data from label file as 2D array

            # Flatten list
            for i in y[0]:
                y_loc_temporary.append(i)  # Flatten   2D array to 1D array

            # Read target label
            with open('Laurence_Data/Main_Task/' + list_of_main_tasks[session] + '/Labels_target.csv',
                      'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                y = []
                for row in reader:
                    int_row = list(map(float, row))  # Map String to float per 1D array
                    y.append(int_row)  # Store data from label file as 2D array

            # Flatten list
            for i in y[0]:
                y_tar_temporary.append(i)  # Flatten   2D array to 1D array

            # Add container lists to their corresponding parent lists
            self.X.append(X_temporary)
            self.y_cue.append(y_cue_temporary)
            self.y_loc.append(y_loc_temporary)
            self.y_tar.append(y_tar_temporary)

    def print_length(self):
        """Debugging function that prints length of data"""
        print("Predictor data (X)")
        print("Number of sessions: \t{}".format(len(self.X)))
        print("Number of observations in Session 0: \t{}".format(len(self.X[0])))
        print("Number of features in Session 0: \t\t{}".format(len(self.X[0][0])))
        print("\nLabel data (y)")
        print("Number of sessions for cue label: \t{}".format(len(self.y_cue)))
        print("Number of observations for cue label in Session 0: {}".format(len(self.y_cue[0])))
        print()     # Print newline



# Divide data into 9 sessions
        # Number of sessions

    # Divide each sector into 45 epochs of corresponding lengths
        # Number of epochs
        # Number of neurons per observation

    # Parameter search for SVM

    def implement_svm(self, split=0.2, verbose = True, parameter_search_option=True, scale_data=True,
                      normalize_data=True):
        """Return raw accuracy values of SVM implementation
        Nested method in method 'save_svm_output_as_csv()' """
        # Do a parameter search if parameter_search_option == true
            # For cue
            # For location
            # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Filename for CSV file
            filename = NeuronDecoding.return_filename(self, algorithm='svm', scale_data=scale_data,
                                                      normalize_data=normalize_data,
                                                      parameter_search_option=parameter_search_option)

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])               # Use numpy to enable multi-dimensional slicing
            C_param, gamma_param, kernel_param = 1, 1, 'poly'    # Set defaults for C, gamma and kernel parameters

            for epoch in range(45):
                X_main = []      # all of X data for corresponding epoch
                X_main_scaled = []      # scaled X data
                X_main_norm = []        # normalized X data
                np.savetxt("matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end], delimiter=',')     # Save X of epoch as csv output
                with open("matrix_test_epoch_"+str(session)+".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row)) # Map String to float per 1D array
                        X_main.append(int_row)     # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                # If True for parameter search
                if parameter_search_option:
                    C_param, gamma_param, kernel_param = NeuronDecoding.parameter_search_svm(self,
                                                                                             X_train_data=X_main,
                                                                                             y_train_data=
                                                                                             self.y_cue[session])
                    print("Parameter search complete for cue.")
                    print("C: {}\tgamma: {}\tkernel: {}".format(C_param, gamma_param, kernel_param))

                # For cue
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split, random_state=42) # Split data
                clf = SVC(C=C_param, gamma=gamma_param, kernel=kernel_param) # Fit SVC() model with parameters
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_cue.append(clf.score(X_test, y_test))    # Add container list to session list
                cue_score = clf.score(X_test, y_test)                               # for verbose

                # If True for parameter search
                if parameter_search_option:
                    C_param, gamma_param, kernel_param = NeuronDecoding.parameter_search_svm(self, X_train_data=X_main,
                                                                                             y_train_data=self.y_loc[session])
                    print("Parameter search complete for location.")
                    print("C: {}\tgamma: {}\tkernel: {}".format(C_param, gamma_param, kernel_param))

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split, random_state=42) # Split data
                clf = SVC(C=C_param, gamma=gamma_param, kernel=kernel_param) # Fit SVC() model model with parameters
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))    # Add container list to session list
                loc_score = clf.score(X_test, y_test)                       # for verbose

                # If True for parameter search
                if parameter_search_option:
                    C_param, gamma_param, kernel_param = NeuronDecoding.parameter_search_svm(self, X_train_data=X_main,
                                                                                             y_train_data=self.y_tar[session])
                    print("Parameter search complete for target.")
                    print("C: {}\tgamma: {}\tkernel: {}".format(C_param, gamma_param, kernel_param))

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split, random_state=42) # Split data
                clf = SVC(C=C_param, gamma=gamma_param, kernel=kernel_param) # Fit SVC() model model with parameters
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))    # Add container list to session list
                tar_score = clf.score(X_test, y_test)                       # for target score

                if verbose:
                    print("SVM: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]), len(self.y_tar[session])))
                    print("Cue score:\t\t{}".format(cue_score))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("main_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("main_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("main_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("main_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar       # Return 2D accuracy lists

    def calculate_average(self, results):
        """Compute average
        results: 2D array of raw data"""

        averaged_results = []
        for num_col in range(len(results[0])):
            temp = 0.0
            for num_row in range(len(results)):
                print(num_col, num_row)
                temp += (results[num_row][num_col])
            averaged_results.append(temp / len(results))

        return averaged_results

    def save_mnb_output_as_csv(self):
        """Save MNB output as CSV files with averaged results"""
        # Call MNB method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronDecoding.implement_mnb(self)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('mnb_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('mnb_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('mnb_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("MNB output saved as CSV files")

    def save_knn_output_as_csv(self, normalize_data=False, scale_data=False, parameter_search_option=False):
        """Save KNN output as CSV files with averaged results"""
        # Call KNN method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronDecoding.implement_knn(self, scale_data=scale_data, normalize_data=normalize_data, parameter_search_option=parameter_search_option)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('knn_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('knn_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('knn_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("KNN output saved as CSV files")

    def save_mlp_output_as_csv(self):
        """Save MLP output as CSV files with averaged results"""
        # Call MLP method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronDecoding.implement_mlp(self)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('mlp_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('mlp_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('mlp_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("MLP output saved as CSV files")

    def save_gnb_output_as_csv(self, scale_data=True, normalize_data=True):
        """Save GNB output as CSV files with averaged results"""
        # Call GNB method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronDecoding.implement_gnb(self, scale_data=scale_data, normalize_data=normalize_data)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('gnb_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('gnb_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('gnb_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("GNB output saved as CSV files")


    def save_svm_output_as_csv(self, normalize_data=True, scale_data=True, parameter_search_option=True, verbose=True):
        """Save SVM output as CSV files with averaged results"""
        # Call SVM method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronDecoding.implement_svm(self,
                        normalize_data=normalize_data, scale_data=scale_data, parameter_search_option=parameter_search_option,
                                                                                                  verbose=verbose)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('svm_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('svm_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('svm_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("SVM output saved as CSV files")

    def plot_output(self,plot_svm=False,plot_gnb=False,plot_knn=False,plot_mlp=False,verbose=False):
        """Import code from files
        plot_svm = True     means that the method will plot svm output
        plot_x = True       means that the method will plot x output
        verbose = True      means that the method will print in the console the data used
        """

        y_cue_average = []  # List with average for cue in 1 dimension
        y_loc_average = []  # List with average for location in 1 dimension
        y_tar_average = []  # List with average for target in 1 dimension

        # For SVM:
        if plot_svm:
            with open("svm_accuracy_cue.csv","r") as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                y = []                  # Temporary list with 2 dimensions
                for row in reader:
                    int_row = list(map(float, row))  # Map String to float per 1D array
                    y_cue_average.append(int_row)   # Store data from label file as 2D array
                # Flatten list
                for i in y:
                    y_cue_average.append(i)  # Flatten   2D array to 1D array

            with open("svm_accuracy_loc.csv","r") as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                y = []                  # Temporary list with 2 dimensions
                for row in reader:
                    int_row = list(map(float, row))  # Map String to float per 1D array
                    y_loc_average.append(int_row)   # Store data from label file as 2D array

            with open("svm_accuracy_tar.csv","r") as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                y = []                  # Temporary list with 2 dimensions
                for row in reader:
                    int_row = list(map(float, row))  # Map String to float per 1D array
                    y_loc_average.append(int_row)   # Store data from label file as 2D array

            x_axis = [i for i in range(45)]     # Generate x axis

            if verbose:                      # For debugging purposes
                print("Data used for graph\n")
                print("X axis: {}".format(x_axis))
                print()
                print("Cue: {}".format(y_cue_average))
                print()

            ax = plt.subplot(111)
            # 'fig' and 'ax' from matplotlib

            plt.plot(x_axis,y_cue_average)# .plot() method
            # save as jpg
            plt.show()

            # For GNB:
            if plot_gnb:
                with open("gnb_accuracy_cue.csv", "r") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    y = []  # Temporary list with 2 dimensions
                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        y_cue_average.append(int_row)  # Store data from label file as 2D array
                    # Flatten list
                    for i in y:
                        y_cue_average.append(i)  # Flatten   2D array to 1D array

                with open("gnb_accuracy_loc.csv", "r") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    y = []  # Temporary list with 2 dimensions
                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        y_loc_average.append(int_row)  # Store data from label file as 2D array

                with open("gnb_accuracy_tar.csv", "r") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    y = []  # Temporary list with 2 dimensions
                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        y_loc_average.append(int_row)  # Store data from label file as 2D array

                x_axis = [i for i in range(45)]  # Generate x axis

                if verbose:  # For debugging purposes
                    print("Data used for graph\n")
                    print("X axis: {}".format(x_axis))
                    print()
                    print("Cue: {}".format(y_cue_average))
                    print()

                ax = plt.subplot(111)
                # 'fig' and 'ax' from matplotlib

                plt.plot(x_axis, y_cue_average)  # .plot() method
                # save as jpg
                plt.show()

    # Implement MNB
    def implement_mnb(self, split=0.2, verbose=True, parameter_search_option=False, scale_data=False,
                      normalize_data=False):
        """Return raw accuracy values of MNB (Multinomial Naive Bayes) implementation"""
        # Do a parameter search if parameter_search_option == true
            # For cue
            # For location
            # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Filename for each iteration
            filename = NeuronDecoding.return_filename(self, algorithm='mnb', scale_data=scale_data,
                                                      normalize_data=normalize_data,
                                                      parameter_search_option=parameter_search_option)
            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])       # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []      # all of X data for corresponding epoch
                np.savetxt("matrix_test_epoch_" +str(session)+".csv", X_session[:, start:end], delimiter=',')     # Save X of epoch as csv output
                with open("matrix_test_epoch_"+str(session)+".csv") as csvfile:
                    reader = csv.reader(csvfile,delimiter = ',')

                    for row in reader:
                        int_row = list(map(float, row)) # Map String to float per 1D array
                        X_main.append(int_row)     # Store data from predictor file as 2D array


                # For cue
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split, random_state=42) # Split data
                clf = MultinomialNB() # Fit MNB() model
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_cue.append(clf.score(X_test, y_test))    # Add container list to session list
                cue_score = clf.score(X_test, y_test)

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split, random_state=42) # Split data
                clf = MultinomialNB() # Fit MNB() model
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))    # Add container list to session list
                loc_score = clf.score(X_test, y_test)

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split, random_state=42) # Split data
                clf = MultinomialNB() # Fit MNB() model
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))    # Add container list to session list
                tar_score = clf.score(X_test, y_test)

                if verbose == True:
                    print("GNB: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]), len(self.y_tar[session])))
                    print("Cue Score:\t\t{}".format(cue_score))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("main_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("main_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("main_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("main_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar       # Return 2D accuracy lists

    # Implement MLP
    def implement_mlp(self, split=0.2, verbose=True,parameter_search_option=False,
                      normalize_data=False, scale_data=False):
        """Return raw accuracy values of MLP (Multilayer perceptron) implementation"""
        # Do a parameter search if parameter_search_option == true
            # For cue
            # For location
            # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Filename for each iteration
            filename = NeuronDecoding.return_filename(self, algorithm='mlp', scale_data=scale_data,
                                                      normalize_data=normalize_data,
                                                      parameter_search_option=parameter_search_option)
            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])       # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []      # all of X data for corresponding epoch
                np.savetxt("matrix_test_epoch_" +str(session)+".csv", X_session[:, start:end], delimiter=',')     # Save X of epoch as csv output
                with open("matrix_test_epoch_"+str(session)+".csv") as csvfile:
                    reader = csv.reader(csvfile,delimiter = ',')

                    for row in reader:
                        int_row = list(map(float, row)) # Map String to float per 1D array
                        X_main.append(int_row)     # Store data from predictor file as 2D array


                # For cue
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split, random_state=42) # Split data
                mlp = MLPClassifier(hidden_layer_sizes=(54, 54, 54), max_iter=1500, activation="logistic")  # Fit MLP() model
                mlp.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_cue.append(mlp.score(X_test, y_test))    # Add container list to session list
                cue_score = mlp.score(X_test, y_test)

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split, random_state=42) # Split data
                mlp = MLPClassifier(hidden_layer_sizes=(54, 54, 54), max_iter=1500, activation="logistic")  # Fit MLP() model
                mlp.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_loc.append(mlp.score(X_test, y_test))    # Add container list to session list
                loc_score = mlp.score(X_test, y_test)

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split, random_state=42) # Split data
                mlp = MLPClassifier(hidden_layer_sizes=(54, 54, 54), max_iter=1500, activation="logistic")  # Fit MLP() model
                mlp.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_tar.append(mlp.score(X_test, y_test))    # Add container list to session list
                tar_score = mlp.score(X_test, y_test)

                if verbose == True:
                    print("MLP: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]), len(self.y_tar[session])))
                    print("Cue Score:\t\t{}".format(cue_score))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("main_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("main_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("main_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("main_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar       # Return 2D accuracy lists

    # Implement GNB
    def implement_gnb(self, split=0.2, verbose=True, parameter_search_option=False, scale_data=False, normalize_data=False):
        """Return raw accuracy values of GNB (Gaussian Naive Bayes) implementation"""
        # Do a parameter search if parameter_search_option == true
        # For cue
        # For location
        # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Filename for each iteration
            filename = NeuronDecoding.return_filename(self, algorithm='gnb', scale_data=scale_data,
                                                      normalize_data=normalize_data,
                                                      parameter_search_option=parameter_search_option)

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])  # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []  # all of X data for corresponding epoch
                X_main = []      # all of X data for corresponding epoch
                X_main_scaled = []      # scaled X data
                X_main_norm = []        # normalized X data
                np.savetxt("matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end],
                           delimiter=',')  # Save X of epoch as csv output
                with open("matrix_test_epoch_" + str(session) + ".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        X_main.append(int_row)  # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session,
                                                                    epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                # For cue
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split,
                                                                    random_state=42)  # Split data
                clf = GaussianNB()  # Fit MNB() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_cue.append(clf.score(X_test, y_test))  # Add container list to session list
                cue_score = clf.score(X_test, y_test)

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split,
                                                                    random_state=42)  # Split data
                clf = GaussianNB()  # Fit MNB() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))  # Add container list to session list
                loc_score = clf.score(X_test, y_test)

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split,
                                                                    random_state=42)  # Split data
                clf = GaussianNB()  # Fit MNB() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))  # Add container list to session list
                tar_score = clf.score(X_test, y_test)

                if verbose == True:
                    print("GNB: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]), len(self.y_tar[session])))
                    print("Cue Score:\t\t{}".format(cue_score))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("main_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("main_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("main_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("main_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar  # Return 2D accuracy lists

    # Implement ANN with Keras
    def implement_keras_ann(self, split=0.2, verbose=True, parameter_search_option=False,
                      normalize_data=False, scale_data=False):
        """Neural network implementation with Keras"""
       # Do a parameter search if parameter_search_option == true
        # For cue
        # For location
        # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Filename for each iteration
            filename = NeuronDecoding.return_filename(self, algorithm='keras_ann', scale_data=scale_data,
                                                      normalize_data=normalize_data,
                                                      parameter_search_option=parameter_search_option)

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])  # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []  # all of X data for corresponding epoch
                X_main_scaled = []  # scaled X data
                X_main_norm = []  # normalized X data
                np.savetxt("matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end],
                           delimiter=',')  # Save X of epoch as csv output
                with open("matrix_test_epoch_" + str(session) + ".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        X_main.append(int_row)  # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                # For cue
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                         X_train_data=X_main,
                                                                                         y_train_data=
                                                                                         self.y_cue[session])
                    print("Parameter search complete for cue.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))

                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,
                                                     algorithm=algorithm)  # Fit KNN() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_cue.append(clf.score(X_test, y_test))  # Add container list to session list
                cue_score = clf.score(X_test, y_test)

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                         X_train_data=X_main,
                                                                                         y_train_data=
                                                                                         self.y_loc[session])
                    print("Parameter search complete for location.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,
                                                     algorithm=algorithm)  # Fit KNN() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))  # Add container list to session list
                loc_score = clf.score(X_test, y_test)

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                         X_train_data=X_main,
                                                                                         y_train_data=
                                                                                         self.y_tar[session])
                    print("Parameter search complete for target.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,
                                                     algorithm=algorithm)  # Fit KNN() modelclf = neighbors.KNeighborsClassifier()          # Fit KNN() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))  # Add container list to session list
                tar_score = clf.score(X_test, y_test)

                if verbose:
                    print("KNN: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]),
                                                            len(self.y_tar[session])))
                    print("Cue Score:\t\t{}".format(cue_score))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("main_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("main_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("main_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("main_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar  # Return 2D accuracy lists

    # Implement KNN
    def implement_knn(self, split=0.2, verbose=True, parameter_search_option=False,
                      normalize_data=False, scale_data=False, n_neighbors=5, metric='minkowski', algorithm='auto'):
        """Return raw accuracy values of GNB (Gaussian Naive Bayes) implementation"""
        # Do a parameter search if parameter_search_option == true
        # For cue
        # For location
        # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Filename for each iteration
            filename = NeuronDecoding.return_filename(self, algorithm='knn', scale_data=scale_data,
                                                      normalize_data=normalize_data,
                                                      parameter_search_option=parameter_search_option)

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])  # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []  # all of X data for corresponding epoch
                X_main_scaled = []  # scaled X data
                X_main_norm = []  # normalized X data
                np.savetxt("matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end],
                           delimiter=',')  # Save X of epoch as csv output
                with open("matrix_test_epoch_" + str(session) + ".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        X_main.append(int_row)  # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                # For cue
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                                 X_train_data=X_main,
                                                                                                 y_train_data=
                                                                                                 self.y_cue[session])
                    print("Parameter search complete for cue.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))

                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)          # Fit KNN() model
                clf.fit(X_train, y_train)                       # Add output score for each epoch to container list
                container_accuracy_total_cue.append(clf.score(X_test, y_test))  # Add container list to session list
                cue_score = clf.score(X_test, y_test)



                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                                 X_train_data=X_main,
                                                                                                 y_train_data=
                                                                                                 self.y_loc[session])
                    print("Parameter search complete for location.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)          # Fit KNN() model
                clf.fit(X_train, y_train)                       # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))  # Add container list to session list
                loc_score = clf.score(X_test, y_test)


                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                                 X_train_data=X_main,
                                                                                                 y_train_data=
                                                                                                 self.y_tar[session])
                    print("Parameter search complete for target.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)          # Fit KNN() modelclf = neighbors.KNeighborsClassifier()          # Fit KNN() model
                clf.fit(X_train, y_train)                       # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))  # Add container list to session list
                tar_score = clf.score(X_test, y_test)

                if verbose:
                    print("KNN: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]),
                                                            len(self.y_tar[session])))
                    print("Cue Score:\t\t{}".format(cue_score))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("main_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("main_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("main_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("main_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar  # Return 2D accuracy lists

    # Parameter search for SVM
    def parameter_search_svm(self, X_train_data, y_train_data, C=[0.1,1,10,100], gamma=[1e-5,1e-3,1e-1,1e1], kernel=['linear','poly','rbf'], verbose=False):
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
        return clf.best_estimator_.C, clf.best_estimator_.gamma, clf.best_estimator_.kernel

    # Parameter search for KNN
    def parameter_search_knn(self, X_train_data, y_train_data, n_neighbors=[5,8,10,20,30,50,100],
                             metric=["minkowski", "manhattan", "euclidean"], algorithm = ['auto'], verbose=False):
        """Parameter search for SVM. Returns best parameters in form: C, gamma, kernel
        X_train_data: X
        y_train_data: y
        Default parameters are arguments"""
        parameters = [{'n_neighbors':n_neighbors, 'metric': metric, 'algorithm': algorithm}]
        knn = neighbors.KNeighborsClassifier()
        clf = GridSearchCV(knn, parameters)
        clf.fit(X_train_data, y_train_data)     # Fit model
        if verbose:   # If verbose, export parameters
            means = clf.cv_results_['mean_test_score']
            params = clf.cv_results_['params']
            for mean, param in zip(means, params):
                print("{}\t{}".format(round(mean, 3), param))

        # Return best parameters in list form
        return clf.best_estimator_.n_neighbors, clf.best_estimator_.metric, clf.best_estimator_.algorithm

    # Normalize data
    def normalize_data(self, X_train_data, y_train_data=None, session=0, epoch=0, verbose=True):
        """Normalize data for any dataset
        X_train_data: X
        y_train_data: y"""
        X_train_norm = normalize(X_train_data)         # Fit data
        if verbose:             # If verbose, export data under CSV format
            np.savetxt("normalized_data_for_session_"+str(session)+"_epoch_"+str(epoch)+".csv", X_train_data, delimiter=',')
        return X_train_norm     # Return normalized data

    # Scale data
    def scale_data_max_abs(self, X_train_data, y_train_data=None, session=0, epoch=0, verbose=True):
        """Scale data with max abs for any dataset
        X_train_data: X
        y_train_data: y"""
        max_abs_scaler = MaxAbsScaler()
        X_train_max_abs = max_abs_scaler.fit_transform(X_train_data)        # Fit data
        if verbose:             # If verbose, export data under CSV format
            np.savetxt("normalized_data_for_session_" + str(session) + "_epoch_" + str(epoch) + ".csv", X_train_max_abs, delimiter=',')
        return X_train_max_abs  # Return scaled data

    # Return filename
    def return_filename(self, algorithm="", scale_data=False, normalize_data=False, parameter_search_option=False):
        """Method that returns filename based on run options, e.g.
        * scale data
        * normalize data
        * parameter search"""
        filename="Invalid filename"
        try:
            filename = algorithm            # if works, filename starts with algorithm name
            if scale_data:
                filename = filename + "_" + "scale"
            if normalize_data:
                filename = filename + "_" + "norm"
            if parameter_search_option:
                filename = filename + "_" + "psearch"
        except:      # If parameters don't work
            print("Invalid parameters for filename!")
        filename = filename + "_"
        return filename

    # Plot data

class NeuronTwistTask(NeuronDecoding):
    """Child class of NeuronDecoding to be used only for Twist Task instead of Main Task
    because there is no Cue label"""
    def __init__(self, X=None, y_cue=None, y_loc=None, y_tar=None, neurons_per_session=None, number_of_epochs=None):
        NeuronDecoding.__init__(self)
        self.neurons_per_session = [146, 133, 148, 146, 129, 132, 136, 141, 131]    # Number of neurons for Twist Task

    def read_global_data_twist(self, number_of_sessions=9,
                               list_of_main_tasks=['K070317', 'K100317', 'K120317', 'K130317', 'K150317', 'K160317',
                                                   'K210217', 'K230217', 'K240217']):
        """Read data from source for TWIST TASK"""
        # Source pathname
        # All predictor (X) data and all label (y) data
        # Add each session as a list in self.X and self.y_cue, self.y_loc, self.y_tar
        for session in range(number_of_sessions):
            # Temporary container variable
            X_temporary = []
            y_cue_temporary = []
            y_loc_temporary = []
            y_tar_temporary = []

            # Read data from corresponding folder

            # Open predictor file
            with open(str('Laurence_Data/Twist_Task/' + list_of_main_tasks[session] + '/Data_matrix.csv'),'r') as csvfile:
                # Read predictor file
                reader = csv.reader(csvfile, delimiter=',') # Read CSV file using ',' as delimiter

                for row in reader:
                    int_row = list(map(float, row)) # Map String to float per 1D array
                    X_temporary.append(int_row)     # Store data from predictor file as 2D array

            try:
                # Read cue label
                with open('Laurence_Data/Twist_Task/'+ list_of_main_tasks[session] + '/Labels_cue.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    y = []
                    for row in reader:
                        int_row = list(map(float, row)) # Map String to float per 1D array
                        y.append(int_row)               # Store data from label file as 2D array

                # Flatten list
                for i in y[0]:
                    y_cue_temporary.append(i)                        # Flatten   2D array to 1D array
            except:
                print("No cue label.")

            # Read location label
            with open('Laurence_Data/Twist_Task/' + list_of_main_tasks[session] + '/Lbales_location.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                y = []
                for row in reader:
                    int_row = list(map(float, row))  # Map String to float per 1D array
                    y.append(int_row)                # Store data from label file as 2D array

            # Flatten list
            for i in y[0]:
                y_loc_temporary.append(i)  # Flatten   2D array to 1D array

            # Read target label
            with open('Laurence_Data/Twist_Task/' + list_of_main_tasks[session] + '/Labels_target.csv',
                      'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                y = []
                for row in reader:
                    int_row = list(map(float, row))  # Map String to float per 1D array
                    y.append(int_row)  # Store data from label file as 2D array

            # Flatten list
            for i in y[0]:
                y_tar_temporary.append(i)  # Flatten   2D array to 1D array

            # Add container lists to their corresponding parent lists
            self.X.append(X_temporary)
            try:
                self.y_cue.append(y_cue_temporary)
            except:
                pass
            self.y_loc.append(y_loc_temporary)
            self.y_tar.append(y_tar_temporary)

    def implement_svm(self, split=0.2, verbose = True, parameter_search_option=True, scale_data=True,
                      normalize_data=True):
        """Return raw accuracy values of SVM implementation
        Nested method in method 'save_svm_output_as_csv()'
        For Twist method only"""
        print("SVM METHOD FOR TWIST TASK.")
        # Do a parameter search if parameter_search_option == true
            # For cue
            # For location
            # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []
        filename = 'DEFAULT'            # Debugging purposes if NeuronDecoding.return_filename() method fails

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])               # Use numpy to enable multi-dimensional slicing
            C_param, gamma_param, kernel_param = 1, 1, 'poly'    # Set defaults for C, gamma and kernel parameters

            filename = NeuronDecoding.return_filename(self, algorithm = 'svm', scale_data=scale_data, normalize_data=normalize_data,
                                       parameter_search_option=parameter_search_option)

            for epoch in range(45):
                X_main = []      # all of X data for corresponding epoch
                X_main_scaled = []      # scaled X data
                X_main_norm = []        # normalized X data
                np.savetxt("twist_matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end], delimiter=',')     # Save X of epoch as csv output
                with open("twist_matrix_test_epoch_"+str(session)+".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row)) # Map String to float per 1D array
                        X_main.append(int_row)     # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                try:
                    # For cue
                    X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split, random_state=42) # Split data
                    clf = SVC(C=C_param, gamma=gamma_param, kernel=kernel_param) # Fit SVC() model with parameters
                    clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                    container_accuracy_total_cue.append(clf.score(X_test, y_test))    # Add container list to session list
                    cue_score = clf.score(X_test, y_test)                               # for verbose

                except:
                    print("No cue label")

                # If True for parameter search
                if parameter_search_option:
                    C_param, gamma_param, kernel_param = NeuronDecoding.parameter_search_svm(self, X_train_data=X_main,
                                                                                             y_train_data=self.y_loc[session])
                    print("Parameter search complete for location.")
                    print("C: {}\tgamma: {}\tkernel: {}".format(C_param, gamma_param, kernel_param))


                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split, random_state=42) # Split data
                clf = SVC(C=C_param, gamma=gamma_param, kernel=kernel_param) # Fit SVC() model model with parameters
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))    # Add container list to session list
                loc_score = clf.score(X_test, y_test)                       # for verbose

                # If True for parameter search
                if parameter_search_option:
                    C_param, gamma_param, kernel_param = NeuronDecoding.parameter_search_svm(self, X_train_data=X_main,
                                                                                             y_train_data=self.y_tar[session])
                    print("Parameter search complete for target.")
                    print("C: {}\tgamma: {}\tkernel: {}".format(C_param, gamma_param, kernel_param))

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split, random_state=42) # Split data
                clf = SVC(C=C_param, gamma=gamma_param, kernel=kernel_param) # Fit SVC() model model with parameters
                clf.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))    # Add container list to session list
                tar_score = clf.score(X_test, y_test)                       # for target score

                if verbose:
                    print("SVM: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]), len(self.y_tar[session])))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("twist_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
            np.savetxt("twist_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

            # After computing 45 epochs, add container lists to total accuracy list
            accuracy_total_cue.append(container_accuracy_total_cue)
            accuracy_total_loc.append(container_accuracy_total_loc)
            accuracy_total_tar.append(container_accuracy_total_tar)

        # After computing for 45 epochs, export accuracy as CSV file
        np.savetxt("twist_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
        np.savetxt("twist_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar       # Return 2D accuracy lists

    # Implement GNB
    def implement_gnb(self, split=0.2, verbose=True, parameter_search_option=False, scale_data=False, normalize_data=False):
        """Return raw accuracy values of GNB (Gaussian Naive Bayes) implementation for Twist task"""
        print("GNB for Twist Task")
        # Do a parameter search if parameter_search_option == true
        # For cue
        # For location
        # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # Filename for each iteration
        filename = NeuronDecoding.return_filename(self, algorithm='gnb', scale_data=scale_data, normalize_data=normalize_data,
                                   parameter_search_option=parameter_search_option)

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])  # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []  # all of X data for corresponding epoch
                X_main = []      # all of X data for corresponding epoch
                X_main_scaled = []      # scaled X data
                X_main_norm = []        # normalized X data
                np.savetxt("twist_matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end],
                           delimiter=',')  # Save X of epoch as csv output
                with open("twist_matrix_test_epoch_" + str(session) + ".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        X_main.append(int_row)  # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session,
                                                                    epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                # For cue
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split,
                                                                        random_state=42)  # Split data
                    clf = GaussianNB()  # Fit MNB() model
                    clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                    container_accuracy_total_cue.append(clf.score(X_test, y_test))  # Add container list to session list
                    cue_score = clf.score(X_test, y_test)
                except:
                    print("No cue label")

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split,
                                                                    random_state=42)  # Split data
                clf = GaussianNB()  # Fit MNB() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))  # Add container list to session list
                loc_score = clf.score(X_test, y_test)

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split,
                                                                    random_state=42)  # Split data
                clf = GaussianNB()  # Fit MNB() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))  # Add container list to session list
                tar_score = clf.score(X_test, y_test)

                if verbose:
                    print("GNB: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]), len(self.y_tar[session])))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("twist_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("twist_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("twist_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("twist_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar  # Return 2D accuracy lists

    def save_svm_output_as_csv(self, normalize_data=True, scale_data=True, parameter_search_option=True, verbose=True):
        """TWIST: Save SVM output as CSV files with averaged results"""
        # Call SVM method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronTwistTask.implement_svm(self,
                        normalize_data=normalize_data, scale_data=scale_data, parameter_search_option=parameter_search_option,
                                                                                                  verbose=verbose)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('twist_svm_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('twist_svm_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('twist_svm_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("Twist Task SVM output saved as CSV files")

    def save_gnb_output_as_csv(self, scale_data=False, normalize_data=False):
        """TWIST: Save GNB output as CSV files with averaged results"""
        # Call GNB method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronTwistTask.implement_gnb(self, scale_data=scale_data, normalize_data=normalize_data)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('twist_gnb_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('twist_gnb_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('twist_gnb_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("Twist Task GNB output saved as CSV files")

    # Implement KNN
    def implement_knn(self, split=0.2, verbose=True, parameter_search_option=False,
                      normalize_data=False, scale_data=False, n_neighbors=5, metric='minkowski', algorithm='auto'):
        """Return raw accuracy values of GNB (Gaussian Naive Bayes) implementation"""
        print("Twist Task for KNN")

        # Do a parameter search if parameter_search_option == true
        # For cue
        # For location
        # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        # Filename for each iteration
        filename = NeuronDecoding.return_filename(self, algorithm='knn', scale_data=scale_data, normalize_data=normalize_data,
                                   parameter_search_option=parameter_search_option)

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])  # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []  # all of X data for corresponding epoch
                X_main_scaled = []  # scaled X data
                X_main_norm = []  # normalized X data
                np.savetxt("twist_matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end],
                           delimiter=',')  # Save X of epoch as csv output
                with open("twist_matrix_test_epoch_" + str(session) + ".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        X_main.append(int_row)  # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session,
                                                                    epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                # For cue
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split,
                                                                        random_state=42)  # Split data
                    # If True for parameter search
                    if parameter_search_option:
                        n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                             X_train_data=X_main,
                                                                                             y_train_data=
                                                                                             self.y_cue[session])
                        print("Parameter search complete for cue.")
                        print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))

                    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,
                                                         algorithm=algorithm)  # Fit KNN() model
                    clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                    container_accuracy_total_cue.append(clf.score(X_test, y_test))  # Add container list to session list
                    cue_score = clf.score(X_test, y_test)
                except:
                    print("No cue label.")

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                         X_train_data=X_main,
                                                                                         y_train_data=
                                                                                         self.y_loc[session])
                    print("Parameter search complete for location.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,
                                                     algorithm=algorithm)  # Fit KNN() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_loc.append(clf.score(X_test, y_test))  # Add container list to session list
                loc_score = clf.score(X_test, y_test)

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split,
                                                                    random_state=42)  # Split data
                # If True for parameter search
                if parameter_search_option:
                    n_neighbors, metric, algorithm = NeuronDecoding.parameter_search_knn(self,
                                                                                         X_train_data=X_main,
                                                                                         y_train_data=
                                                                                         self.y_tar[session])
                    print("Parameter search complete for target.")
                    print("Neighbors: {}\tMetric: {}\tAlgorithm: {}".format(n_neighbors, metric, algorithm))
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,
                                                     algorithm=algorithm)  # Fit KNN() modelclf = neighbors.KNeighborsClassifier()          # Fit KNN() model
                clf.fit(X_train, y_train)  # Add output score for each epoch to container list
                container_accuracy_total_tar.append(clf.score(X_test, y_test))  # Add container list to session list
                tar_score = clf.score(X_test, y_test)

                if verbose:
                    print("KNN: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]),
                                                            len(self.y_tar[session])))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("twist_loc_" + filename + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("twist_tar_" + filename + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_cue.append(container_accuracy_total_cue)
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("twist_final_loc_" + filename + ".csv", accuracy_total_loc, delimiter=',')
            np.savetxt("twist_final_tar_" + filename + ".csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar  # Return 2D accuracy lists

    def save_mlp_output_as_csv(self, normalize_data=False, scale_data=False, parameter_search_option=False):
        """Save MLP output as CSV files with averaged results"""
        # Call MLP method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronTwistTask.implement_mlp(self)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('twist_mlp_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('twist_mlp_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('twist_mlp_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("Twist Task MLP output saved as CSV files")

    def save_knn_output_as_csv(self, normalize_data=False, scale_data=False, parameter_search_option=False):
        """Save KNN output as CSV files with averaged results"""
        # Call KNN method
        accuracy_total_cue, accuracy_total_loc, accuracy_total_tar = NeuronTwistTask.implement_knn(self, scale_data=scale_data, normalize_data=normalize_data, parameter_search_option=parameter_search_option)

        averaged_cue = NeuronDecoding.calculate_average(self, accuracy_total_cue)
        averaged_loc = NeuronDecoding.calculate_average(self, accuracy_total_loc)
        averaged_tar = NeuronDecoding.calculate_average(self, accuracy_total_tar)

        # Save as CSV files
        np.savetxt('twist_knn_accuracy_cue.csv', averaged_cue, delimiter=',')
        np.savetxt('twist_knn_accuracy_loc.csv', averaged_loc, delimiter=',')
        np.savetxt('twist_knn_accuracy_tar.csv', averaged_tar, delimiter=',')
        print("Twist Task KNN output saved as CSV files")

    # Implement MLP
    def implement_mlp(self, split=0.2, verbose=True, scale_data=False, normalize_data=False, parameter_search_option=False):
        """Return raw accuracy values of MLP (Multilayer perceptron) implementation"""
        # Do a parameter search if parameter_search_option == true
            # For cue
            # For location
            # For target
        # Accuracy lists
        accuracy_total_cue = []
        accuracy_total_loc = []
        accuracy_total_tar = []

        print("Twist Task for MLP")

        # For each of the 9 sessions
        for session in range(9):
            # For each of the 45 epoch
            # Container lists for the accuracy
            container_accuracy_total_cue = []
            container_accuracy_total_loc = []
            container_accuracy_total_tar = []

            # Intervals of X for all 45 epochs
            interval = self.neurons_per_session[session]
            start = 0
            end = interval
            X_session = np.array(self.X[session])       # Use numpy to enable multi-dimensional slicing

            for epoch in range(45):
                X_main = []  # all of X data for corresponding epoch
                X_main_scaled = []  # scaled X data
                X_main_norm = []  # normalized X data
                np.savetxt("twist_matrix_test_epoch_" + str(session) + ".csv", X_session[:, start:end],
                           delimiter=',')  # Save X of epoch as csv output
                with open("twist_matrix_test_epoch_" + str(session) + ".csv") as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')

                    for row in reader:
                        int_row = list(map(float, row))  # Map String to float per 1D array
                        X_main.append(int_row)  # Store data from predictor file as 2D array

                # If True for scaling data:
                if scale_data:
                    X_main_scaled = NeuronDecoding.scale_data_max_abs(self, X_main, session=session, epoch=epoch)
                    if normalize_data:
                        X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session,
                                                                    epoch=epoch)
                        X_main = X_main_norm
                        print("Data scaled and normalized.")

                # If just True for normalizing data
                if normalize_data and scale_data == False:
                    X_main_norm = NeuronDecoding.normalize_data(self, X_main_scaled, session=session, epoch=epoch)
                    X_main = X_main_norm
                    print("Data normalized and not scaled.")

                try:
                    # For cue
                    X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_cue[session], test_size=split, random_state=42) # Split data
                    mlp = MLPClassifier(hidden_layer_sizes=(54, 54, 54), max_iter=1500, activation="logistic")  # Fit MLP() model
                    mlp.fit(X_train,y_train)    # Add output score for each epoch to container list
                    container_accuracy_total_cue.append(mlp.score(X_test, y_test))    # Add container list to session list
                    cue_score = mlp.score(X_test, y_test)
                except:
                    print("No cue label.")

                # For location
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_loc[session], test_size=split, random_state=42) # Split data
                mlp = MLPClassifier(hidden_layer_sizes=(54, 54, 54), max_iter=1500, activation="logistic")  # Fit MLP() model
                mlp.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_loc.append(mlp.score(X_test, y_test))    # Add container list to session list
                loc_score = mlp.score(X_test, y_test)

                # For target
                X_train, X_test, y_train, y_test = train_test_split(X_main, self.y_tar[session], test_size=split, random_state=42) # Split data
                mlp = MLPClassifier(hidden_layer_sizes=(54, 54, 54), max_iter=1500, activation="logistic")  # Fit MLP() model
                mlp.fit(X_train,y_train)    # Add output score for each epoch to container list
                container_accuracy_total_tar.append(mlp.score(X_test, y_test))    # Add container list to session list
                tar_score = mlp.score(X_test, y_test)

                if verbose:
                    print("MLP: Session {}\tEpoch\t{}".format(session, epoch))
                    print("START: {}\t\tEND:\t{}".format(start, end))
                    print("Label length: {}\t{}\t{}".format(len(self.y_cue[session]), len(self.y_loc[session]), len(self.y_tar[session])))
                    print("Location score:\t\t{}".format(loc_score))
                    print("Target score:\t\t{}".format(tar_score))
                    print()

                # Update interval
                start = end
                end += interval

                # After computing for 45 epochs, export accuracy as CSV file
                np.savetxt("twist_mlp_loc_session_" + str(session) + ".csv", container_accuracy_total_loc, delimiter=',')
                np.savetxt("twist_mlp_tar_session_" + str(session) + ".csv", container_accuracy_total_tar, delimiter=',')

                # After computing 45 epochs, add container lists to total accuracy list
                accuracy_total_loc.append(container_accuracy_total_loc)
                accuracy_total_tar.append(container_accuracy_total_tar)

            # After computing for 45 epochs, export accuracy as CSV file
            np.savetxt("twist_final_mlp_loc_session.csv", accuracy_total_loc, delimiter=',')
            np.savetxt("twist_final_mlp_tar_session.csv", accuracy_total_tar, delimiter=',')

        return accuracy_total_cue, accuracy_total_loc, accuracy_total_tar       # Return 2D accuracy lists

if __name__ == "__main__":

    model = NeuronDecoding()            # Use NeuronDecoding class
    model.read_global_data()            # Read all data necessary
    model.print_length()                # Print lengths
    model.save_knn_output_as_csv(parameter_search_option=True, scale_data=True, normalize_data=True)      # Save as CSV files
