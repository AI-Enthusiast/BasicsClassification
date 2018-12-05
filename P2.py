# P2.py started on 11/13/2018
# Author: Cormac Dacker (cxd289)
# Date: 6 December 2018

import random

import keras
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split


# Used to self report errors
def err(errorMessage):
    print("> ERROR:\t" + errorMessage)


def sigmoid(z):
    return 1 / (1 + pd.np.exp(-z))


def c(w1=-1, w2=-1, yIntcpt=6.465, prob = "1.C: Decision boundary "):  # plots the decition boundary with weights set by hand
    # alternatively the user could be prompted for these
    # y = - w1 * x + w2 * x / w3

    # gathered from part a

    color = (data[["species"]].values.T).astype("uint8")  # how to color the point on the scatter plot
    plt.scatter(x[50:], y[50:], c=color[0, 50:], s=40)

    # Draws the desision boundary
    X, Y = list(range(1, 3)), []  # what will be the line
    for index in range(len(X)):
        Y.append((-w1/w2) * X[index] + yIntcpt)  # creating the line
    plt.plot(X, Y)
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title(prob + str(w1) + ", " + str(w2) + ", " + str(yIntcpt))
    plt.show()


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

learningRate = .001


# Uses Logistic regression and returns the weights for prob 2 to use
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
        def e():  # Part E, calls Prob 2
            rows = [53, 57, 56, 70, 77, 136, 119, 106, 128, 136]
            x = list(data.iloc[rows, [2, 3]].values)  # Length and Width data
            y = list(data.iloc[rows, 4].values)  # class data
            return train(x, y, "e")

        def predict(p):
            return 1 if p >= .5 else 0

        def sigmoidDeriv(T):
            try:
                return T * (1 - T)
            except RuntimeWarning as e:
                print(sigmoid)
                err(e)

        def ErResWeiDerive(T, t, sigmoid, data):  # derivative of the error with respect to the weight
            return (T - t) * sigmoidDeriv(sigmoid) * data  # sigmoid is the last input to the last neuron

        def error(T, t):  # prob 3
            return ((T - t))

        def train(data, labels, prob='b'):  # Trains the modle based off the given training
            def test(data=None, labels=None, prob="b"):  # Part D is inbedded to use info from part B
                def d():  # Part D
                    w1Val = pd.np.arange(1,2.5, 1.5/100)
                    w2Val = pd.np.arange(3,7,4/100)
                    Z = pd.np.zeros([100,100])
                    for i in range(len(w1Val)): # plots the Z using w1val and w2vals as x&y
                        for j in range(len(w2Val)): # maps x and y to the z axis
                            Z[i][j]= (sigmoid(w1Val[i] * w1 + w2Val[j] * w2 + bias))

                    # for 3d plotting
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    Z = pd.np.array(Z).T
                    Axes3D.plot_wireframe(self = ax, X=pd.np.array(w1Val), Y = pd.np.array(w2Val), Z =Z)

                    plt.xlabel("w1")
                    plt.ylabel("w2")
                    plt.title("1.D: Output of the sigmoid over the input space")
                    plt.show()

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

                if prob == 'b':  # check if it's 1.b
                    c()  # call part C
                    d()  # call part D
                    e()  # call part E
                    return (w1, w2, bias)

                # print(" y = " + str(-w1) + "/" + str(w2) + " * x" + " - " + str(bias))
                # print("Prob of Error  =", str(100 * numWrong / len(x_test)) + "%")

            # initializing variables for loop
            w1, w2, bias = .5, .5, -3.5  # initialize weights and biases
            w1Update, w2Update, biasUpdate = 0, 0, 0
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
                totErr.append(error(T, t))
                # print("(x,y)=", (x_train[index], t))
            #     w1Update += (T - t) * sigmoidDeriv(sigmoid(priorN))  # update w2
            #     w2Update += (T - t) * sigmoidDeriv(sigmoid(priorN))  # update w2
            #     biasUpdate += (T - t) * sigmoidDeriv(sigmoid(N))
            #
            # w1 = w1 - - learningRate * data[index][0]  # update w2
            # w2 = w2 - - learningRate * data[index][1]  # update w2
            # bias = bias - learningRate * biasUpdate  # update bias

            return test(data, labels, prob)

        return train(x_train, y_train)

    a()  # run part A
    return b()  # run part B, calls part C, D, and E


# Applies mean squared error
def prob2(w):
    def se(y, probY): #Squared error
        return (probY - y) ** 2

    def mseModel():
        results = []
        for i in range(len(xV)):
            guesses = []
            N = (xV[i][0] * w[0] + xV[i][1] * w[1] + w[2])
            guesses.append(sigmoid(N))
            results.append(se(yV[i], guesses[i]))
        print("MSE =" + str(pd.np.mean(results)) + "%")

    print("\nPROB 2.A:")
    mseModel()

    print("\nPROB 2.B:")
    w = [.5, .6, 6.45]
    mseModel()
    c(w[0],w[1],w[2], prob="1.B: Decision boundary ")
    w = [-50, -50, -.5]
    mseModel()
    c(w[0],w[1],w[2], prob="1.B: Decision boundary ")



# Uses a tool library to make it ez
def prob4():
    def modelNN(dim):  # the model being used
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
    prob2(prob1())  # run problem 1
    # prob4()
