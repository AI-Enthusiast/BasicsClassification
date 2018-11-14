# P2E1.py started on 11/13/2018
# Authors: Cormac Dacker
# Excersise 1

import csv
import matplotlib.pyplot as plt

def error(errorMessage):
    print("> ERROR:\t" + errorMessage)


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
        data = readFile("irisdata.csv")
        out= []
        for index in range(len(data)): # data seperates x(length) and y(width) axis for the 2&3 classes
            if data[index][4] == "virginica" or data[index][4] == "versicolor":
                out.append([data[index][2],data[index][3]])
        out.sort()
        for index in range(len(out)):
            plt.plot(out[index][0],out[index][1], 'ro')
        plt.show()
    def b():
        print("TEMP")
    a()

if __name__ == "__main__":
    problem1()