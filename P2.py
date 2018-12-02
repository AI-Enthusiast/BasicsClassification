# P2.py started on 11/13/2018
# Author: Cormac Dacker (cxd289)
# Date: 6 December 2018

import math
import random

import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Used to self report errors
def err(errorMessage):
    print("> ERROR:\t" + errorMessage)


# Problem 1 of the homework #TODO CLean up before submitting
data = pd.read_csv("irisdata.csv")

# reassigns species labels
data.loc[data['species'] == "setosa", "species"] = 0
data.loc[data['species'] == "versicolor", "species"] = 1
data.loc[data['species'] == "virginica", "species"] = 2

x = data["petal_width"].values.T  # x data for scatter plot
y = data["petal_length"].values.T  # y data for scatter plot

# split into relevent data sections
xV = list(data.iloc[50:, [2, 3]].values)  # Length and Width data
yV = list(data.iloc[50:, 4].values)  # class data

# slit into testing and training
x_train, x_test, y_train, y_test = train_test_split(xV, yV, test_size=0.20, random_state=1)


# Uses Logistic regression
def prob1():
    def a():  # Scatter plot of petal width and length of classes 2&3

        color = (data[["species"]].values.T).astype("uint8")  # how to color the point on the scatter plot
        plt.scatter(x[50:], y[50:], c=color[0, 50:],
                    s=40)  # create a scatter plot of only the 2&3classes (row 50 and down)

        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.title("1.A: Iris Data | 2nd & 3rd Classes")
        plt.show()  # ggez

    def b():  # Part B, contains Part D and E
        def e():  # Part E, calls Prob 1
            rows = [53, 57, 56, 70, 77, 136, 119, 106, 128, 136]
            x = list(data.iloc[rows, [2, 3]].values)  # Length and Width data
            y = list(data.iloc[rows, 4].values)  # class data
            train(x, y, "e")

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

        def train(data, labels, prob='b'):  # Trains the modle based off the given training
            def test(data=None, labels=None, prob="b"):  # Part D is inbedded to use info from part B
                errRate = []
                numWrong = 0
                if prob == 'b':  # check if it's 1.b
                    data = x_test
                    labels = y_test
                    print("PROB 1.B:")
                else:
                    print("PROB 1.E:")
                # tests the data
                for index in range(len(data)):

                    actual = labels[index] - 1  # true class
                    predicted = predict(sigmoid(data[index][0] * w1 + data[index][1] * w2 + bias))  # predicted class
                    out = "√"
                    if predicted != actual:
                        numWrong += 1
                        out = "X"
                    errRate.append(((index + 1) - numWrong) / (index + 1))
                    print("Predicted =", predicted, ", Actual =", actual, "Result =", out)
                print("Score = ", str(100 * (len(data) - numWrong) / len(data)) + "%\n  ")

                def d():  # Part D
                    plt.plot(list(range(len(x_test))), errRate)
                    plt.xlabel("Iteration")
                    plt.ylabel("Proportion Correct on Test Set")
                    plt.title("1.D: Error Rate per Generation")
                    plt.show()

                if prob == 'b':  # check if it's 1.b
                    c()  # call part C
                    d()  # call part D
                    e()  # call part E
                # print(" y = " + str(-w1) + "/" + str(w2) + " * x" + " - " + str(bias))
                # print("Prob of Error  =", str(100 * numWrong / len(x_test)) + "%")

            # initializing variables for loop
            w1, w2, bias = random.random(), random.random(), -random.random()  # initialize weights and biases
            totErr = []  # the total error
            N, priorN = 0, 0
            for index in range(len(data)):
                # print(x_train[index][0] , w1 , x_train[index][1] , w2 , bias)
                priorN = N
                N = data[index][0] * w1 + data[index][1] * w2 + bias  # N = length * w1 + width * w2 + bias
                # print("N",N) #temp
                # print("W1, w2= " , w1, w2)
                T = sigmoid(N)  # 1-0 on what class it thinks it is
                t = (labels[index] - 1)  # minus one bc 2nd and 3rd classes = 1 & 2
                totErr.append(error(T, t))  # ?????????????
                # print("(x,y)=", (x_train[index], t))

                w1 = w1 - (T - t) * sigmoidDeriv(sigmoid(priorN)) * .1 * data[index][0]  # update w2
                w2 = w2 - (T - t) * sigmoidDeriv(sigmoid(priorN)) * .1 * data[index][1]  # update w2
                m = -w1 / w2  # the slope
                bias = bias - (T - t) * sigmoidDeriv(sigmoid(N))  # update bias

            test(data, labels, prob)

        train(x_train, y_train)

    def c():  # plots the decition boundary with weights set by hand
        w1 = -1  # alternatively the user could be prompted for these
        w2 = -1
        m = -w1 / w2
        bias = 6.465  # acting as the x intercept when it should be y intercept

        # gathered from part a
        x = data["petal_width"].values.T  # x data for scatter plot
        y = data["petal_length"].values.T  # y data for scatter plot
        color = (data[["species"]].values.T).astype("uint8")  # how to color the point on the scatter plot
        plt.scatter(x[50:], y[50:], c=color[0, 50:], s=40)

        # Draws the desision boundary
        X, Y = list(range(1, 3)), []  # what will be the line
        for index in range(len(X)):
            Y.append(m * X[index] + bias)  # creating the line
        plt.plot(X, Y)
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.title("1.C: Decision boundary " + str(w1) + ", " + str(w2) + ", " + str(bias))
        plt.show()

    a()  # run part A
    b()  # run part B, calls part C, D, and E


def prob2():
    def mse(y, probY):
        return pd.np.mean((y - probY) ** 2)


# Uses a tool library to make it ez
def prob4():
    def modelNN(dim): # the model being used
        model = keras.Sequential()
        model.add(keras.layers.Dense(1, input_dim=dim, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def a():  # Part A, following the tutorial
        print("\nPROB 4.A:")
        model = modelNN(2)
        model.fit(x=pd.np.array(x_train), y=pd.np.array(y_train), epochs=1000,
                  validation_data=(pd.np.array(x_test), pd.np.array((y_test))))

    def b():  # Part B, the complete iris data
        print("\nPROB 4.B:")

        # split into relevent data sections
        xS = list(data.iloc[0:, [0, 1, 2, 3]].values)  # Length and Width data
        yS = list(data.iloc[0:, 4].values)  # class data

        # slit into testing and training
        x_train, x_test, y_train, y_test = train_test_split(xS, yS, test_size=0.20, random_state=1)

        model = modelNN(4)
        model.fit(x=pd.np.array(x_train), y=pd.np.array(y_train), epochs=1000,
                  validation_data=(pd.np.array(x_test), pd.np.array((y_test))))

    a()  # run part A
    b()  # run part B


if __name__ == "__main__":
    prob1()  # run problem 1
    prob4()
