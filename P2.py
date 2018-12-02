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
    data = pd.read_csv("irisdata.csv")

    # reassigns species labels
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

        def d():  # Part D is inbedded to use info from part B
            errRate = []
            numWrong = 0
            # tests the data
            for index in range(len(x_test)):
                actual = y_test[index] - 1  # true class
                predicted = predict(sigmoid(x_test[index][0] * w1 + x_test[index][1] * w2 + bias))  # predicted class
                if predicted != actual:
                    numWrong += 1
                errRate.append(((index + 1) - numWrong )/ (index + 1))
                print("Predicted =", predicted, ", Actual =", actual)
            # print(" y = " + str(-w1) + "/" + str(w2) + " * x" + " - " + str(bias))
            # print("Prob of Error  =", str(100 * numWrong / len(x_test)) + "%")
            print(errRate)
            plt.plot(list(range(len(x_test))), errRate)
            plt.xlabel("Iteration")
            plt.ylabel("Proportion Correct on Test Set")
            plt.title("Error Rate per Generation")
            plt.show()
            print("Score = ", (len(x_test) - numWrong)/len(x_test))

        def predict(p):
            return 1 if p >= .5 else 0

        def sigmoid(z):
            return 1 / (1 + math.exp(-z))

        def sigmoidDeriv(T):
            try:
                return T * (1 - T)
            except RuntimeWarning as e:
                print(sigmoid)
                err(e)

        def ErResWeiDerive(T, t, sigmoid, data):  # derivative of the error with respect to the weight
            return (T - t) * sigmoidDeriv(sigmoid) * data  # sigmoid is the last input to the last neuron

        def error(T, t):
            return .5 * ((T - t) ** 2)

        # split into relevent data sections
        x = list(data.iloc[50:, [2, 3]].values)  # Length and Width data
        y = list(data.iloc[50:, 4].values)  # class data
        # slit into testing and training
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

        # initializing variables for loop
        w1, w2, bias = random.random(), random.random(), -random.random()  # initialize weights and biases
        totErr = []  # the total error
        N, priorN = 0, 0

        # Trains the modle based off the given training data
        for index in range(len(x_train)):
            # print(x_train[index][0] , w1 , x_train[index][1] , w2 , bias)
            priorN = N
            N = x_train[index][0] * w1 + x_train[index][1] * w2 + bias  # N = length * w1 + width * w2 + bias
            # print("N",N) #temp
            # print("W1, w2= " , w1, w2)
            T = sigmoid(N)  # 1-0 on what class it thinks it is
            t = (y_train[index] - 1)  # minus one bc 2nd and 3rd classes = 1 & 2
            totErr.append(error(T, t)) #?????????????
            # print("(x,y)=", (x_train[index], t))

            w1 = w1 - (T - t) * sigmoidDeriv(sigmoid(priorN)) * .1 * x_train[index][0]  # update w2
            w2 = w2 - (T - t) * sigmoidDeriv(sigmoid(priorN)) * .1 * x_train[index][1]  # update w2
            m = -w1 / w2  # the slope
            bias = bias - (T - t) * sigmoidDeriv(sigmoid(N))  # update bias

        #print(totErr)

        #c(bias)
        d()

    def c(bias):  # plots the decition boundary with weights set by hand
        # set weights by hand
        w1 = -1  # alternatively the user could be prompted fro these
        w2 = -1
        m = -w1 / w2
        bias = 6.465  # acting as the x intercep when i should be y intercept
        # gathered from part a
        x = data["petal_width"].values.T  # x data for scatter plot
        y = data["petal_length"].values.T  # y data for scatter plot
        color = (data[["species"]].values.T).astype("uint8")  # how to color the point on the scatter plot
        plt.scatter(x[50:], y[50:], c=color[0, 50:],
                    s=40)

        # Draws the desision boundary
        X, Y = list(range(1, 3)), []  # what will be the line
        for index in range(len(X)):
            Y.append(m * X[index] + bias)  # creating the line
        plt.plot(X, Y)
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.title("Desition boundary " + str(w1) + ", " + str(w2) + ", " + str(bias))
        plt.show()

    def e():
        pass

    #a()  # run part A
    b()  # run part B


if __name__ == "__main__":
    prob1()  # run problem 1
