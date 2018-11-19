# P2E1.py started on 11/13/2018
# Authors: Cormac Dacker
# Excersise 1

import csv
import pandas as pd
import matplotlib.pyplot as plt

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
        #createFile().close()
        return start

#Problem 1 of the homework
def problem1():
    def a(): # Part A
        data = pd.read_csv("irisdata.csv")

        # reassigns lables
        data.loc[data['species'] == "setosa", "species"] = 0
        data.loc[data['species'] == "versicolor", "species"] = 1
        data.loc[data['species'] == "virginica", "species"] = 2

        x = data["petal_width"].values.T # x data for scatter plot
        y = data["petal_length"].values.T # y data for scatter plot
        color = (data[["species"]].values.T).astype("uint8") # how we color the point on the scatter plot
        plt.scatter(x,y, c=color[0,:], s = 40)
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.title("Iris Data | 2nd & 3rd Classes")
        plt.show()

    def b():
        print("TEMP")
    a()

if __name__ == "__main__":
    problem1()

