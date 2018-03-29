from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition


import csv
import time
import matplotlib.pyplot as plt

import Neuron_V4_file as NV4


neuron_obj = NV4.NeuronDecoding()
neuron_obj.read_global_data()

class ModelKeras:
    """Class used for running ANNs"""

    def __init__(self, n_tasks=469, n_epochs=136, session=0):
        self.n_tasks = n_tasks
        self.n_epochs = n_epochs
        self.session = session
        self.input_index = [[10,12,15,16,45,67,74,115],
                       [4,17,57,62,71,107,115,122,130,132,136,137,158],
                       [5,15,29,43,67,81,98,110,115,116,118],
                       [1,2,7,14,18,33,45,57,81,114,115,119,120,121,122,124,127,133,140,142,153],
                       [49,81,95,108,111,112,115,118,128],
                       [1,21,58,65,66,74,81,82,99,104,117,122,127,135,151,155,167],
                       [1,61,66,81,107,112,113,114,122,123,124,127,133],
                       [1,3,43,45,61,67,68,73,81,89,103,116,120,121,123,124,125,128,133,138,140],
                       [1,3,4,29,62,67,71,75,76,83,116,121,128,132,134,135,139,144]
        ]

        self.output_index = [[5,25,26,31,35,36,37,40,50,55,58,61,62,68,69,71,73,78,79,80,81,86,92,94,99,105,107,120,123,125,133,135],
                        [8,9,11,12,13,15,18,21,24,25,27,29,30,31,33,34,40,42,43,44,45,48,51,66,67,69,74,76,79,82,83,84,85,86,89,90,91,92,93,94,95,97,100,103,104,105,108,112,118,120,121,123,124,128,129,133,138,140,145,146,147,149,151,152,153,154,155,163,164,166,167],
                        [1,4,10,12,14,19,23,26,28,31,32,33,39,41,47,48,52,54,58,68,70,74,76,77,78,79,82,84,85,87,90,92,94,96,105,106,108,129,130,136,140],
                        [4,5,6,9,12,13,17,19,22,23,24,29,32,34,38,39,40,42,51,64,67,68,69,70,73,77,79,80,82,83,85,87,88,89,92,94,95,96,97,99,103,106,107,108,112,125,128,129,131,134,136,143,145,146,147,149,150,151,154,155,157,160],
                        [3,7,11,12,16,19,20,21,23,24,29,30,31,33,36,37,39,42,44,45,46,47,54,59,61,64,67,71,72,79,82,83,84,85,86,87,90,91,93,100,102,103,106,107,114,117,119,120,121,127,131,136,137,139,140,143,144,146],
                        [4,5,7,10,11,15,18,20,22,23,24,28,31,34,36,42,45,46,47,50,51,52,54,61,64,68,77,78,79,83,84,85,86,89,90,91,92,93,94,95,96,100,101,102,105,112,113,114,115,118,120,121,125,126,128,129,130,133,139,140,142,146,147,154,156,157,158,159,161,162,163,164,165,169,170,172,174],
                        [5,8,9,10,11,13,15,18,20,22,24,27,28,29,34,38,40,41,43,45,46,47,67,68,72,75,77,83,84,85,87,88,89,90,91,93,94,96,98,102,109,137,139,140,141,142,143,146,148,150],
                        [12,14,16,17,22,23,25,28,33,38,40,41,48,50,51,53,54,55,58,69,78,83,90,91,97,99,104,106,109,132,141,142,150,153,154,155,156,159,161,167,168],
                        [8,9,15,16,23,24,25,26,28,31,32,35,42,44,46,47,48,49,50,51,52,55,56,58,59,60,61,65,66,79,80,84,85,86,89,92,93,94,96,98,101,102,104,105,111,115,122,124,126,129,130,147,149,152,154,155,156,159,161,162,168]
        ]

        self.nb_tasks=[469, 301, 399, 318, 384, 249, 269, 288, 280]

        self.nb_neurons=[136, 167, 140, 160, 147, 174, 150, 168, 168]

    def input_or_output(self, CONSTANT, normalized_data):
        """Determines whether neuron is input neuron or output neuron"""
        input_neurons, output_neurons = [], []
        CONSTANT = CONSTANT*self.nb_neurons[self.session]    # CONSTANT is the epoch number

        # for number of tasks
        for test in range(self.n_tasks):
            temp_input = []
            temp_output = []
            # for the number of neurons in that task
            for index in range(self.nb_neurons[self.session]):
                if index in self.input_index[self.session]:
                    temp_input.append(normalized_data[test][CONSTANT + index-1])
                if index in self.output_index[self.session]:
                    temp_output.append(normalized_data[test][CONSTANT + index-1])
            input_neurons.append(temp_input)
            output_neurons.append(temp_output)
        return input_neurons, output_neurons

    def split_target_1_2(self, input_neurons, output_neurons, n_tasks=469):
        """Split data according to target 1 or 2"""
        y_target = neuron_obj.y_tar

        # Separate into 1 and 2
        data_tar_in_1, data_tar_in_2 = [], []
        data_tar_out_1, data_tar_out_2 = [], []

        for test in range(self.nb_tasks[self.session]):
            if y_target[self.session][test] == 1:
                data_tar_in_1.append(input_neurons[test])
                data_tar_out_1.append(output_neurons[test])
            if y_target[self.session][test] == 2:
                data_tar_in_2.append(input_neurons[test])
                data_tar_out_2.append(output_neurons[test])

        return data_tar_in_1, data_tar_in_2, data_tar_out_1, data_tar_out_2

    def flatten_weights(self, model_weights):
        flatened_weights = []
        for array in model_weights:
            array = array.flatten()
            for item in array:
                flatened_weights.append(item)
        return flatened_weights

    def run_ann_target1(self, data_tar_in_1, data_tar_out_1, activation_var='sigmoid', loss_var='mean_squared_error',
        optimizer_var='adam', metrics_var=['acc'], epoch_var=50, batch_size_var=30, n_tasks=469):
        weights_tar_1= []
        start = 0
        end = 15
        interval = int(self.nb_tasks[self.session]/(end*2))       # number of networks
        for test in range(interval):

            X = np.array(data_tar_in_1[start:end])
            y = np.array(data_tar_out_1[start:end])
            np.random.seed(7)
            print(len(X))
            print(len(y))
            r_model = self.model(X, y, activation_var=activation_var, loss_var=loss_var, optimizer_var=optimizer_var, metrics_var=metrics_var, epoch_var=epoch_var,
             batch_size_var=batch_size_var)
            temp = r_model.get_weights()
            weights_tar_1.append(self.flatten_weights(temp))
            start += interval
            end += interval
            print(test)
            del temp
            del r_model
        return weights_tar_1
        del weights_tar_1

    def model(self, X, y, activation_var='sigmoid', loss_var='mean_squared_error', optimizer_var='Adam', metrics_var=['acc'], epoch_var=50, batch_size_var=30):
        start = time.time()

        # Build neural network
        # Sequential
        model = Sequential()

        # Neural network
        model.add(Dense(5, input_dim=len(X[0]), activation='sigmoid' ))
        model.add(Dense(5, activation='sigmoid' ))
        model.add(Dense(len(y[0]), activation='sigmoid' ))


        # Compile model
        # sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
        model.compile(loss=loss_var, optimizer=optimizer_var, metrics=metrics_var)


        # Fit model
        history = model.fit(X, y, validation_split=0.2, nb_epoch=epoch_var, batch_size=batch_size_var, verbose=0)

        end = time.time()

        elapsed = end - start

        print("ELAPSED TIME:", elapsed)
        # print(model.predict([[1.,2.,10.]]))

        """
        # Analysis
        # Plot data
                # Using matplotlib, plot success rate (2 axis chart with selected data)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel("Epoch")
        plt.legend(['train', 'test'], loc='lower right')

        # summarize history for loss
        plt.subplot(212)v
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss (%)')
        plt.xlabel("Epoch")
        plt.legend(['train', 'test'], loc='lower left')
        plt.savefig('keras_test.png', dpi=100)
        plt.show()
        """
        return model
        del model

    def run_ann_target2(self, data_tar_in_2, data_tar_out_2, activation_var='sigmoid', loss_var='mean_squared_error',
        optimizer_var='adam', metrics_var=['acc'], epoch_var=50, batch_size_var=30, n_tasks=469):
        """Returns weights 2"""
        weights_tar_2= []

        start = 0
        end = 15
        interval = int(self.nb_tasks[self.session]/(end*2))       # number of networks
        for test in range(interval):

            X = np.array(data_tar_in_2[start:end])
            y = np.array(data_tar_out_2[start:end])
            np.random.seed(7)
            r_model = self.model(X, y, activation_var=activation_var, loss_var=loss_var, optimizer_var=optimizer_var, metrics_var=metrics_var, epoch_var=epoch_var,
             batch_size_var=batch_size_var)
            temp = r_model.get_weights()
            weights_tar_2.append(self.flatten_weights(temp))
            start += interval
            end += interval
            print(test)
            del r_model
            del temp
        return weights_tar_2
        del weights_tar_2


    def merge_lists_split(self, weights_tar_1, weights_tar_2, split=0.3, random_num=42):
        X_main = []
        y_main = []
        # Merge lists 1 and 2 together
        for item in weights_tar_1:
            X_main.append(item)
            y_main.append(1)
        for item in weights_tar_2:
            X_main.append(item)
            y_main.append(2)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size = split, random_state = random_num)
        return X_train, X_test, y_train, y_test

    def implement_svm(self, X_train, X_test, y_train, y_test):
        # Parameter search
        C=[0.1,1,10,100]
        gamma=[1e-5,1e-3,1e-1,1e1]
        kernel=['linear','poly','rbf']


        parameters = [{'C': C, 'gamma': gamma, 'kernel': kernel}]
        svc = SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train, y_train)     # Fit model

        means = clf.cv_results_['mean_test_score']
        params = clf.cv_results_['params']
        #for mean, param in zip(means, params):
         #   print("{}\t{}".format(round(mean, 3), param))


        clf_svm = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
        clf_svm.fit(X_train, y_train)
        score = clf_svm.score(X_test, y_test)
        del clf_svm
        return(score)


    def run_session_K040217(self, normalized_data, activation_var='sigmoid', loss_var='mean_squared_error',
        optimizer_var='adam', metrics_var=['acc'], epoch_var=50, batch_size_var=30, start=0, end=45):
       # decoding_accuracy_target = []
        for epoch in range(start, end):    # for each fo the 45 epochs:
            input_neurons, output_neurons = self.input_or_output(epoch, normalized_data) # split neurons based on input or output index
            data_tar_in_1, data_tar_in_2, data_tar_out_1, data_tar_out_2 = self.split_target_1_2(input_neurons, output_neurons) # split neurons based on target 1 or 2
            weights_tar_1 = self.run_ann_target1(data_tar_in_1, data_tar_out_1, optimizer_var=optimizer_var, epoch_var=epoch_var, batch_size_var=batch_size_var, loss_var=loss_var)   # compute weights
            weights_tar_2 = self.run_ann_target2(data_tar_in_2, data_tar_out_2, optimizer_var=optimizer_var, epoch_var=epoch_var, batch_size_var=batch_size_var, loss_var=loss_var)   # compute weights

            for weight_num in range(len(weights_tar_1)):
                a = weights_tar_1[weight_num]
                np.savetxt(str(optimizer_var)+"_weight_tar1_"+str(epoch)+"_trial"+str(weight_num)+'_'+str(batch_size_var)+'_'+str(epoch_var)+".csv", a, delimiter=",")

            for weight_num in range(len(weights_tar_2)):
                a = weights_tar_2[weight_num]
                np.savetxt(str(optimizer_var)+"_weight_tar2_"+str(epoch)+"_trial"+str(weight_num)+'_'+str(batch_size_var)+'_'+str(epoch_var)+".csv", a, delimiter=",")

            rand_list=[x*2+42 for x in range(25)]
            decoding_num = [] # list with decoding accuracies

            for num in rand_list:
                X_train, X_test, y_train, y_test = self.merge_lists_split(weights_tar_1, weights_tar_2, split=0.3, random_num=num)  # split data for testing and training
                decoding_num.append(self.implement_svm(X_train, X_test, y_train, y_test))     # add accuracy on 1 run to list
              #  decoding_accuracy_target.append(decoding_num)    # decoding accuracy on SVM

            print(decoding_num)

            sum_of = 0
            for item in decoding_num:
                sum_of += item
                length = len(decoding_num)

            decoding_num=[]
            decoding_num.append(sum_of/length)

            np.savetxt((str(optimizer_var)+"_new_svm_accuracy_"+str(epoch)+'_'+str(batch_size_var)+'_'+str(epoch_var)+".csv"), decoding_num, delimiter=',')   # save as .csv file
            print("RAN "+str(epoch)+" EPOCHS...\n\n\n\n\n")
            print(decoding_num)

            del weights_tar_1, weights_tar_2
        return decoding_accuracy_target

    def run_session(self, normalized_data, activation_var='sigmoid', loss_var='mean_squared_error',
        optimizer_var='adam', metrics_var=['acc'], epoch_var=50, batch_size_var=30, start=0, end=45, session='VOID', n_tasks=469):
       # decoding_accuracy_target = []
        for epoch in range(start, end):    # for each fo the 45 epochs:
            input_neurons, output_neurons = self.input_or_output(epoch, normalized_data) # split neurons based on input or output index
            data_tar_in_1, data_tar_in_2, data_tar_out_1, data_tar_out_2 = self.split_target_1_2(input_neurons, output_neurons) # split neurons based on target 1 or 2
            weights_tar_1 = self.run_ann_target1(data_tar_in_1, data_tar_out_1, optimizer_var=optimizer_var, epoch_var=epoch_var, batch_size_var=batch_size_var, loss_var=loss_var, n_tasks=n_tasks)   # compute weights
            weights_tar_2 = self.run_ann_target2(data_tar_in_2, data_tar_out_2, optimizer_var=optimizer_var, epoch_var=epoch_var, batch_size_var=batch_size_var, loss_var=loss_var, n_tasks=n_tasks)   # compute weights

            # weights for target 1
            for weight_num in range(len(weights_tar_1)):
                a = weights_tar_1[weight_num]
                np.savetxt(str(optimizer_var)+"_weight_tar1_"+str(epoch)+"_trial"+str(weight_num)+'_'+str(batch_size_var)+'_'+str(epoch_var)+'_'+str(self.session)_".csv", a, delimiter=",")

            # weights for target 2
            for weight_num in range(len(weights_tar_2)):
                a = weights_tar_2[weight_num]
                np.savetxt(str(optimizer_var)+"_weight_tar2_"+str(epoch)+"_trial"+str(weight_num)+'_'+str(batch_size_var)+'_'+str(epoch_var)+'_'+str(self.session)".csv", a, delimiter=",")

            rand_list=[x*2+42 for x in range(25)]
            decoding_num = [] # list with decoding accuracies

            for num in rand_list:
                X_train, X_test, y_train, y_test = self.merge_lists_split(weights_tar_1, weights_tar_2, split=0.3, random_num=num)  # split data for testing and training
                decoding_num.append(self.implement_svm(X_train, X_test, y_train, y_test))     # add accuracy on 1 run to list
              #  decoding_accuracy_target.append(decoding_num)    # decoding accuracy on SVM

            print(decoding_num)

            sum_of = 0
            for item in decoding_num:
                sum_of += item
                length = len(decoding_num)

            decoding_num=[]
            decoding_num.append(sum_of/length)

            np.savetxt((str(optimizer_var)+"_new_svm_accuracy_"+str(epoch)+'_'+str(batch_size_var)+'_'+str(epoch_var)+'_'+str(session)+".csv"), decoding_num, delimiter=',')   # save as .csv file
            print("RAN "+str(epoch)+" EPOCHS...\n\n\n\n\n")
            print(decoding_num)

            del weights_tar_1, weights_tar_2


    def calculate_average(self, filename):
        results = []

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for row in reader:
                int_row =[]
                print(row)
                for item in row:
                    try:
                        int_row.append(float(item))
                    except:
                        pass
                results.append(int_row)

        print(results)

        averaged_results=[]
        for num_col in range(len(results[0])):
            temp = 0.0
            for num_row in range(len(results)):
                print(num_col,num_row)
                temp += (results[num_row][num_col])
            averaged_results.append(temp/len(results))

        return averaged_results

    """
    Different parameters
    Different runs on several sessions
    Try the Neuron library"""
