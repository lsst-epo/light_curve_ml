import csv
import os

from astropy.time import Time
import numpy as np
from nupic.algorithms.sdr_classifier import SDRClassifier


_DATA_DIR = "$HOME/PycharmProjects/lsst/data/catalina"


def getDatasetFilePaths(dataDir, datasetName, ext):
    """
    :param dataDir - Directory of datasets
    :param datasetName - Name of dataset
    :param ext - File extension
    """
    path = os.path.join(os.path.expandvars(dataDir), datasetName)
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def toDatetime(time, format="mjd", scale="tt"):
    """Converts time in specified format and scale (e.g, Modified Julian Date
    (MJD) and Terrestrial Time) to datetime."""
    try:
        t = Time(float(time), format=format, scale=scale)
    except ValueError:
        print "Could not create time from: %s" % time
        return None

    return t.datetime


def _extractLightCurveTimeSeries(paths):
    """"""
    results = []
    for path in paths:
        with open(path, "r") as f:
            reader = csv.reader(f)

            # 9 - MJD, 2 - magnitude
            results.append([(toDatetime(row[9]), row[2])
                            for i, row in enumerate(reader) if i])

    return results


def _reportDatasetSize(dataset):
    size = len(dataset)
    dataSizes = [len(x) for x in dataset]
    minSize = min(dataSizes)
    maxSize = max(dataSizes)
    ave = np.average(dataSizes)
    std = np.std(dataSizes)
    print "Dataset size: %s. Items' min: %s ave: %.02f (%.02f) max: %s" % (
        size, minSize, ave, std, maxSize)


def main():
    datasetName = "periodic"
    extension = ".csv"
    paths = getDatasetFilePaths(_DATA_DIR, datasetName, extension)
    dataset = _extractLightCurveTimeSeries(paths)
    _reportDatasetSize(dataset)

    for d in dataset:
        print "\ndt\tmag"
        for t, v in d:
            print "%s\t%s" % (t, v)


def runClassification():
    # http://nupic.docs.numenta.org/stable/api/algorithms/classifiers.html
    """steps - Sequence of the different steps of multi-step predictions
    to learn
    alpha - learning rate (larger -> faster learning)
    actValueAlpha - Used to track the actual value within each bucket.
    A lower actValueAlpha results in longer term memory"""
    c = SDRClassifier(steps=[1], alpha=0.1, actValueAlpha=0.1, verbosity=0)

    # learning
    c.compute(recordNum=0, patternNZ=[1, 5, 9],
              classification={"bucketIdx": 4, "actValue": 34.7},
              learn=True, infer=False)

    # inference
    result = c.compute(recordNum=1, patternNZ=[1, 5, 9],
                       classification={"bucketIdx": 4, "actValue": 34.7},
                       learn=False, infer=True)

    # Print the top three predictions for 1 steps out.
    topPredictions = sorted(zip(result[1],
                                result["actualValues"]), reverse=True)[:3]
    for prob, value in topPredictions:
        print "Prediction of {} has prob: {}.".format(value, prob * 100.0)


if __name__ == "__main__":
    runClassification()
