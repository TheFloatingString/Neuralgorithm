import random
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import csv

from sklearn import decomposition
from sklearn import datasets

class Viz:
    def total_pca(predictor, labels, labeling=2):
        counter=0
        np.random.seed(5)

        fig = plt.figure("Visualisation des neurones")
        ax = fig.add_subplot(111, projection='3d')

        # Rename variables
        X = predictor
        y = labels

        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)

        y_ax_1, y_ax_2, z_ax_1, z_ax_2, x_ax_1, x_ax_2 = [],[],[],[],[],[]
        x_ax_3, y_ax_3, z_ax_3 = [], [], []
        x_ax_4, y_ax_4, z_ax_4 = [], [], []
        x_ax_5, y_ax_5, z_ax_5 = [], [], []
        x_ax_6, y_ax_6, z_ax_6 = [], [], []
        x_ax_7, y_ax_7, z_ax_7 = [], [], []
        x_ax_8, y_ax_8, z_ax_8 = [], [], []


        for i in range(len(X)):
            if labels[i] == 1.0:
                x_ax_1.append(X[i][0])
                y_ax_1.append(X[i][1])
                z_ax_1.append(X[i][2])

            if labels[i] == 2.0:
                x_ax_2.append(X[i][0])
                y_ax_2.append(X[i][1])
                z_ax_2.append(X[i][2])
            counter+=1


            if labels[i] == 3.0:
                x_ax_3.append(X[i][0])
                y_ax_3.append(X[i][1])
                z_ax_3.append(X[i][2])
            counter+=1

            if labels[i] == 4.0:
                x_ax_4.append(X[i][0])
                y_ax_4.append(X[i][1])
                z_ax_4.append(X[i][2])
            counter+=1

            if labels[i] == 5.0:
                x_ax_5.append(X[i][0])
                y_ax_5.append(X[i][1])
                z_ax_5.append(X[i][2])
            counter+=1

            if labels[i] == 6.0:
                x_ax_6.append(X[i][0])
                y_ax_6.append(X[i][1])
                z_ax_6.append(X[i][2])
            counter+=1

            if labels[i] == 7.0:
                x_ax_7.append(X[i][0])
                y_ax_7.append(X[i][1])
                z_ax_7.append(X[i][2])
            counter+=1

            if labels[i] == 8.0:
                x_ax_8.append(X[i][0])
                y_ax_8.append(X[i][1])
                z_ax_8.append(X[i][2])
            counter+=1


        ax.scatter(x_ax_1,y_ax_1,z_ax_1, c='purple', label = "Ensemble de neurones pour cible 1", s=1)
        ax.scatter(x_ax_2,y_ax_2,z_ax_2, c='orange', label = "Ensemble de neurones pour cible 2", s=1)

        if labeling>2:
            ax.scatter(x_ax_3,y_ax_3,z_ax_3, c='red', label = "Ensemble de neurones pour cible 3", s=1)
            ax.scatter(x_ax_4,y_ax_4,z_ax_4, c='blue', label = "Ensemble de neurones pour cible 4", s=1)

            if labeling>4:
                ax.scatter(x_ax_5,y_ax_5,z_ax_5, c='green', label = "Ensemble de neurones pour cible 5", s=1)
                ax.scatter(x_ax_6,y_ax_6,z_ax_6, c='cyan', label = "Ensemble de neurones pour cible 6", s=1)
                ax.scatter(x_ax_7,y_ax_7,z_ax_7, c='brown', label = "Ensemble de neurones pour cible 7", s=1)
                ax.scatter(x_ax_8,y_ax_8,z_ax_8, c='pink', label = "Ensemble de neurones pour cible 8", s=1)

        ax.set_xlabel("Axe x arbitraire")
        ax.set_ylabel("Axe y arbitraire")
        ax.set_zlabel("Axe z arbitraire")

        plt.title("Visualisation du taux de décharge des neurones d'une session")

        if labeling==2:
            plt.legend(loc = "lower left")

        savename = ('3D PCA OF ALL OBSERVATIONS')

        plt.xlim(-600,600)
        plt.ylim(-600,600)

        plt.savefig("Results\\"+savename, dpi=500)
        plt.show()


        import random

    def viz_neurons(X, start=0, end=136, size=469):

            random_numbers = random.sample(range(1, size), 9)
            ordered_nums = [i for i in range(9)]

            for i in range(9):
                num = 330+i+1

                plt.subplot(num)

                x_axis = [x for x in range(45)]


                x_list = []
                for neuron in range(136):
                    temp = []
                    for epoch in range(45):
                        temp.append(X[0+random_numbers[i]][epoch*136 + neuron])
                    x_list.append(temp)

                for j in range(136):
                    plt.plot(x_axis, x_list[j])

                plt.xlabel("Époque")
                plt.ylabel("Décharges")

                if y_tar[random_numbers[i]] == 1:
                    plt.title("Cible 1")

                else:
                    plt.title("Cible 2")


            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.7)
            plt.savefig("Results/VIZ.png", dpi=500)
