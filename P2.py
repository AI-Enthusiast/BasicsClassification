# P2.py started on 11/13/2018
# Author: Cormac Dacker (cxd289)
# Date: 7 December 2018

import random

import keras
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

learningRate = .001 # step size for gradient decent
wBad = [1, 1, 1] # bad weights
wGood = [.2, 3.7, -7] # good weights

data = pd.read_csv("irisdata.csv") # read the data

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


# Used to self report errors
def err(errorMessage):
    print("> ERROR:\t" + errorMessage)


def sigmoid(z):
    return 1 / (1 + pd.np.exp(-z))


# Computes the derivative of a sigmoid
def sigmoidDeriv(T):
    try:
        return T * (1 - T)
    except RuntimeWarning as e:
        print(T)
        err(e)


# Creates a mse model, taking a parameter of weights and retruning the mse for the model and the updated weights
# Only take NN parameters because Iris data is global
def mseModel(w):
    # initialize loop vars
    results = 0  # for the squared error, the nominator of MSE
    w1Update, w2Update, biasUpdate = 0, 0, 0  #

    # Loops though all the Width and length data collecting the Squared error data then the mean of it after the loop
    for i in range(len(xV)):
        N = (xV[i][0] * w[0] + xV[i][1] * w[1] + w[2])  # squish
        guesses = sigmoid(N)  # gives the sigmoid of the squish
        actual = yV[i] - 1  # the actual lable of the class
        results += (guesses - actual) ** 2  # adds the squared error to the results
        w1Update += ((guesses - actual) * sigmoidDeriv(sigmoid(N))) * xV[i][0]  # update w2
        w2Update += ((guesses - actual) * sigmoidDeriv(sigmoid(N))) * xV[i][1]  # update w2
        biasUpdate += (guesses - actual) * sigmoidDeriv(sigmoid(N))  # update bias
    out = results / len(xV)

    newW = list(range(0, 3))  # a list for the new weights
    newW[0] = w[0] - learningRate * w1Update  # update w2
    newW[1] = w[1] - learningRate * w2Update  # update w2
    newW[2] = w[2] - learningRate * biasUpdate  # update bias
    return (out, newW)


# PROB 1C
# plots the decition boundary with weights set by hand
def c(w1, w2, bias, prob):
    color = (data[["species"]].values.T).astype("uint8")  # how to color the point on the scatter plot
    plt.scatter(x[50:], y[50:], c=color[0, 50:], s=40)  # crate a scatter plot of only the 2nd & 3rd classes

    X, Y = list(range(3, 7)), []  # what will be the line
    m = (-w1 / w2)  # finds the slope
    yIntcpt = (-bias) / w2  # converts to the y intercept

    # loop for drawing the line
    for index in range(len(X)):
        Y.append(m * X[index] + yIntcpt)  # creating the line

    # Draws the desision boundary
    plt.plot(Y, X)
    plt.ylabel("Petal Length")
    plt.xlabel("Petal Width")
    plt.title(prob + " Decision boundary ")
    plt.show()


# Uses Logistic regression and returns the weights for prob 2 to use
def prob1():
    def a():  # Scatter plot of petal width and length of classes 2&3
        color = (data[["species"]].values.T).astype("uint8")  # how to color the point on the scatter plot
        plt.scatter(x[50:], y[50:], c=color[0, 50:], s=40)  # crate a scatter plot of only the 2nd & 3rd classes
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.title("1.A: Iris Data | 2nd & 3rd Classes")
        plt.show()  # ggez

    def b():  # Part B, contains Part D and E
        # Part 1E
        def e():
            rows = [53, 57, 56, 70, 77, 136, 119, 106, 128, 136] # fist half are of class 2, second half of class 3
            x = list(data.iloc[rows, [2, 3]].values)  # Length and Width data
            y = list(data.iloc[rows, 4].values)  # class data
            return model(x, y, "e")  # use the modle fuction below

        def model(data=None, labels=None, prob='b'):  # Trains the modle based off the given training
            def test(data=None, labels=None, prob="b"):  # Part D is inbedded to use info from part B
                # Part 1D
                def d():  # create a 3D plot
                    w2Val = pd.np.arange(1, 2.5, 1.5 / 100)  # create data points in the range of the data
                    w1Val = pd.np.arange(3, 7, 4 / 100)
                    Z = pd.np.zeros([100, 100])

                    # loop maps the z axis
                    for i in range(len(w1Val)):  # plots the Z using w1val and w2vals as x&y
                        for j in range(len(w2Val)):  # maps x and y to the z axis
                            Z[j][i] = (
                                sigmoid(w1Val[i] * w1 + w2Val[j] * w2 + bias))  # must be at [j][i] for some reason

                    # for 3d plotting
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    Z = pd.np.array(Z).T  # transpose the Z
                    Axes3D.plot_wireframe(self=ax, X=pd.np.array(w1Val), Y=pd.np.array(w2Val), Z=Z)
                    plt.xlabel("w1")
                    plt.ylabel("w2")
                    plt.title("1.D: Output of the sigmoid over the input space")
                    plt.show()

                if prob == 'b':  # check if it's 1.b
                    data = x_test
                    labels = y_test
                    print("PROB 1.B:")
                else:
                    print("PROB 1.E:")

                # initialize vars before loop
                errRate = []
                numWrong = 0

                # tests the data
                for index in range(len(data)):
                    actual = labels[index] - 1  # true class
                    # prediction of what class it should be
                    predicted = 1 if (sigmoid(data[index][0] * w1 + data[index][1] * w2 + bias)) >= .5 else 0
                    out = "√"
                    if predicted != actual:
                        numWrong += 1
                        out = "X"
                    errRate.append(((index + 1) - numWrong) / (index + 1))
                    print("Predicted =", predicted, ", Actual =", actual, "Result =", out)
                print("Score = ", str(100 * (len(data) - numWrong) / len(data)) + "%\n  ")

                if prob == 'b':  # check if it's 1.b
                    c(wGood[0], wGood[1], wGood[2], "1.C:")  # call part C
                    d()  # call part D
                    e()  # call part E

            # initializing variables for loop in test()
            w1, w2, bias = wGood[0], wGood[1], wGood[2]  # initialize weights and biases
            test(data, labels, prob=prob)

        model(prob="b")

    a()  # run part A
    b()  # run part B


# Applies mean squared error
def prob2():
    # Part 2b
    print("\nPROB 2.B:")  # Shows good and bad MSEs and plots them
    w = wGood
    print("Good: {:.2f}".format(mseModel(w)[0]))
    c(w[0], w[1], w[2], prob="2.B: Good")
    w = wBad
    print("Bad: {:.2f}".format(mseModel(w)[0]))
    c(w[0], w[1], w[2], prob="2.B: Bad")

    # Part 2e
    print("\nPROB 2.E:")  # shows and update with an mse
    w = [.5, .5, -3]
    print("Before: {:.2f}".format(mseModel(w)[0]))
    c(w[0], w[1], w[2], prob="2.E: Before")
    for i in range(0, 500):
        mse = mseModel(w)
    w = mse[1]
    print("After: {:.3f}".format(mseModel(w)[0]))
    c(w[0], w[1], w[2], prob="2.E: After")


# implemets gradient decent using MSE
# PROB 3.A Code
def prob3():
    rng = 10001  # Stop after this many weight adjustments/iterations
    wDifference = list(range(0, rng))  # a placeholder for mse values for plotting
    w = [random.random(), random.random(), -random.random()]  # initialize random weights

    # Loop computes the gradeint decent by updating the weights through the mseModel()
    for i in range(0, rng):
        mse = mseModel(w)  # run the current model
        wDifference[i] = (mse[0])  # adds the mse to the index
        w = mse[1]  # takes the updated weights

        # The if statment below is the code for 3.B
        # plot the DB at 10 iterations into gradient decent, in the middle and 10 before the end.
        if i == 10 or i == int(rng / 2) or i == rng - 10:  # PROB 3.B & C
            mseLst = list(range(0, i))  # this coppies everything from wDifference to the current index
            # This needs to happen because wDifference has a size set to 1000, so plotting without it full is wrong
            for j in range(len(mseLst)):  # does not help with run time  ¯\_(ツ)_/¯ only have to do it 3 times
                mseLst[j] = wDifference[j]

            c(w[0], w[1], w[2], prob="PROB 3.C:")  # plot the decition boundary using 1.c 's code
            plt.plot(list(range(0, len(mseLst))), mseLst)  # Plot the mse curve (should decrease)
            plt.title("PROB 3.C: Learning curve")
            plt.show()
    print(w)


# Uses a tool library to make it ez
def prob4():
    rng = 10  # eazy control over number of epochs

    def a():  # Part A, following the tutorial basically copied, thus no commenting.
        print("\nPROB 4.A:")
        model = keras.Sequential()
        model.add(keras.layers.Dense(1, input_dim=2, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(x=pd.np.array(x_train), y=pd.np.array(y_train), epochs=rng,
                  validation_data=(pd.np.array(x_test), pd.np.array(y_test)))
        results = model.evaluate(pd.np.array(x_test), pd.np.array(y_test))
        print("Accuracy = {:.2f}".format(results[1]))
        print("Loss = {:.2f}".format(results[0]))
        model.summary()

    def b():  # Part B, the complete iris data
        print("\nPROB 4.B:")

        # split into relevent data sections
        xS = list(data.iloc[:, :4].values)  # Length and Width data
        yS = list(data.iloc[:, 4].values)  # class data

        # slit into testing and training
        x_train, x_test, y_train, y_test = train_test_split(xS, yS, test_size=0.20, random_state=1)

        # almost identical to part a
        model = keras.Sequential()
        model.add(keras.layers.Dense(4, input_dim=4, activation='relu'))  # input layer
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(x=pd.np.array(x_train), y=pd.np.array(y_train), epochs=rng,
                  validation_data=(pd.np.array(x_test), pd.np.array(y_test)))
        results = model.evaluate(pd.np.array(x_test), pd.np.array(y_test))
        print("Accuracy = {:.2f}".format(results[1]))
        print("Loss = {:.2f}".format(results[0]))

        model.summary()

    a()  # run part A
    b()  # run part B


if __name__ == "__main__":
    prob1()  # run problem 1
    prob2()  # run problem 2
    prob3()  # run problem 3
    prob4()  # run problem 4
