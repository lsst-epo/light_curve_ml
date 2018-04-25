#!/usr/bin/env python3
"""Adapted from
dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/"""
import time

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from lcml.utils.context_util import joinRoot


def missingFeatures(dataset, description):
    """Requires dataset columns to be unique (could subtract counters"""
    return list(set(dataset) - set(description))


def trainRfClassifier(xTrain, yTrain, numTrees=10, maxFeatures="auto",
                      numJobs=-1):
    """Trains an sklearn.ensemble.RandomForestClassifier.

    :param xTrain: ndarray of features
    :param yTrain: ndarray of labels
    :param numTrees: see n_estimators in
        sklearn.ensemble.forest.RandomForestClassifier
    :param maxFeatures: see max_features in
        sklearn.ensemble.forest.RandomForestClassifier
    :param numJobs: see n_jobs in sklearn.ensemble.forest.RandomForestClassifier
    """
    model = RandomForestClassifier(n_estimators=numTrees,
                                   max_features=maxFeatures, n_jobs=numJobs)
    s = time.time()
    model.fit(xTrain, yTrain)
    print("Trained model in %.2fs", time.time() - s)
    return model


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
    xTrain, xTest, yTrain, yTest = train_test_split(featureHeaders,
                                                    targetHeaders,
                                                    train_size=trainRatio)

    # Train and Test dataset size details
    print("\nTrain & Test sizes")
    print("Train_x Shape: ", xTrain.shape)
    print("Train_y Shape: ", yTrain.shape)
    print("Test_x Shape: ", xTest.shape)
    print("Test_y Shape: ", yTest.shape)

    model = trainRfClassifier(xTrain, yTrain)
    testPredictions = model.predict(xTest)
    trainPredictions = model.predict(xTrain)

    reportSample = 10
    print("\nSample performance")
    t = PrettyTable(["Predicted", "Actual"])

    # convert the dataframe in to list object the indexes will be in order
    testYList = list(yTest)
    for i in range(0, reportSample):
        t.add_row([testYList[i], testPredictions[i]])

    print(t)

    # accuracy
    print("\nFull performance")
    print("Train accuracy: ", accuracy_score(yTrain, trainPredictions))
    print("Test accuracy: ", accuracy_score(yTest, testPredictions))
    print("Confusion: ", confusion_matrix(yTest, testPredictions))


if __name__ == "__main__":
    main()
