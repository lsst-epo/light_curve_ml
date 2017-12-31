"""Adapted from
dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/"""
import os

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from lcml.utils.context_util import joinRoot



def missingFeatures(dataset, description):
    """Requires dataset columns to be unique (could subtract counters"""
    return list(set(dataset) - set(description))


def rfClassifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


def main():
    np.random.seed(0)
    dataPath = joinRoot("data/rf/breast-cancer-wisconsin.csv")
    dataset = pd.read_csv(dataPath)
    headers = list(dataset)
    description = dataset.describe()
    print("Missing features: %s" % missingFeatures(dataset, description))

    # remove missing data
    dataset = dataset[dataset[headers[6]] != "?"]

    # trim away 'CodeNumber' and 'CancerType' columns
    featureHeaders = dataset[headers[1: -1]]
    targetHeaders = dataset[headers[-1]]
    trainRatio = 0.7
    trainX, testX, trainY, testY = train_test_split(featureHeaders,
                                                    targetHeaders,
                                                    train_size=trainRatio)

    # Train and Test dataset size details
    print("\nTrain & Test sizes")
    print("Train_x Shape: ", trainX.shape)
    print("Train_y Shape: ", trainY.shape)
    print("Test_x Shape: ", testX.shape)
    print("Test_y Shape: ", testY.shape)

    model = rfClassifier(trainX, trainY)
    testPredictions = model.predict(testX)
    trainPredictions = model.predict(trainX)

    reportSample = 10
    print("\nSample performance")
    t = PrettyTable(["Predicted", "Actual"])

    # convert the dataframe in to list object the indexes will be in order
    testYList = list(testY)
    for i in range(0, reportSample):
        t.add_row([testYList[i], testPredictions[i]])

    print(t)

    # accuracy
    print("\nFull performance")
    print("Train accuracy: ", accuracy_score(trainY, trainPredictions))
    print("Test accuracy: ", accuracy_score(testY, testPredictions))
    print("Confusion: ", confusion_matrix(testY, testPredictions))


if __name__ == "__main__":
    main()
