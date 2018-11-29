# P2E1.py started on 11/13/2018
# Authors: Cormac Dacker
# Excersise 1

import csv
import math
import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Used to self report errors
def err(errorMessage):
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
    data = pd.read_csv("irisdata.csv")  #

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
        print("temp")

        def predict(p):
            return 1 if p >= .5 else 0

        def sigmoid(z):
            return 1 / (1 + math.exp(-z))

        def sigmoidDeriv(sigmoid):
            try:
                return sigmoid * (1 - sigmoid)
            except RuntimeWarning as e:
                print(sigmoid)
                err(e)

        def ErResWeiDerive(T, t, sigmoid, data):  # derivative of the error with respect to the weight
            return (T + t) * sigmoidDeriv(sigmoid) * data

        def error(T, t):
            return .5 * ((T - t) ** 2)

        x = list(data.iloc[:, [2, 3]].values)  # if all else fails convert it to a string and remove the shit TEMP
        y = list(data.iloc[:, 4].values)
        # slit into testing and training
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
        w1, w2, bias = random.random(), random.random(), -random.random()

        for index in range(len(x_train)):
            N = x_train[index][0] * w1 + x_train[index][0] * w2 + bias
            T = sigmoid(N)
            t = y_train[index]
            print(T, predict(T))
            w1 = w1 - error(T, t) * .1
            w2 = w2 - error(T, t) * .1
            try:
                bias = bias - (T + t) * sigmoidDeriv(N) * .1
            except RuntimeWarning as e:
                print(N)
                err(e)

        print(" y = " + str(-w1 / w2) + " * x" + " - " + str(bias))

    # a()  # run part A
    b()  # run part B


if __name__ == "__main__":
    print("temp")
    prob1()  # run problem 1
