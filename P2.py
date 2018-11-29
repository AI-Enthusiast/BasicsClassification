# P2E1.py started on 11/13/2018
# Authors: Cormac Dacker
# Excersise 1

import csv

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split

import math
# Used to self report errors
def error(errorMessage):
    print("> ERROR:\t" + errorMessage)


# Creates a blank file with the given filename
def createFile(file):
    f = open(file, 'w+', newline='\n', encoding='utf8')
    f.close()


def readFile(file):
    with open(file, "r", newline='', encoding='utf8') as csvfile:
        DataReader = csv.reader(csvfile, delimiter=',', quotechar=" ")
        start, out = [], []
        for item in DataReader:
            start.append(item)
        # createFile().close()
        return start


# Problem 1 of the homework
def prob1():
    data = pd.read_csv("irisdata.csv") #

    # reassigns labels
    data.loc[data['species'] == "setosa", "species"] = 0
    data.loc[data['species'] == "versicolor", "species"] = 1
    data.loc[data['species'] == "virginica", "species"] = 2

    def a():  # Scatter plot of petal width and length of classes 2&3

        x = data["petal_width"].values.T  # x data for scatter plot
        y = data["petal_length"].values.T  # y data for scatter plot
        color = (data[["species"]].values.T).astype("uint8")  # how to color the point on the scatter plot
        plt.scatter(x[50:], y[50:], c=color[0, 50:],
                    s=40)  # create a scatter plot of only the 2&3classes (row 50 and down)

        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.title("Iris Data | 2nd & 3rd Classes")
        plt.show()  # ggez

    def b():
        def predict(p):
            return 1 if p >= .5 else 0

        def logistic(z):
            return ((1/(1 + math.exp(-z))), math.exp(z) + math.exp(-z))

        print(logistic(0))
        x = list(data.iloc[:, [2, 3]].values)  # if all else fails convert it to a string and remove the shit TEMP
        y = list(data.iloc[:, 4].values)
        # slit into testing and training
        # x_train, x_test = (x[50:90,100:140]), (x[90:100],[140:150])
        # y_train, y_test = y[50:90].append(y[100:140]), y[90:100].append(y[140:150])


    # a()  # run part A
    b()  # run part B


if __name__ == "__main__":
    prob1()  # run problem 1
