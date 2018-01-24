import argparse
from collections import Counter
import time

from feets import FeatureSpace
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from lcml.common import STANDARD_INPUT_DATA_TYPES
from lcml.processing.preprocess import cleanDataset
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.context_util import absoluteFilePaths, joinRoot
from lcml.utils.data_util import convertClassLabels, unarchiveAll
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess


logger = getBasicLogger(__name__, __file__)


OGLE3_LABEL_TO_NUM = {'acep': 0, 'cep': 1, 'dpv': 2, 'dsct': 3, 'lpv': 4,
                      'rrlyr': 5}


OGLE3_NUM_TO_LABEL = {v: k for k, v in OGLE3_LABEL_TO_NUM.items()}


def _check_dim(lc):
    if lc.ndim == 1:
        lc.shape = 1, 3
    return lc


def parseOgle3Lc(filePath):
    lc = _check_dim(np.loadtxt(filePath))
    return lc[:, 0], lc[:, 1], lc[:, 2]


def ogle3ToLc(dataDir, limit=float("inf")):
    """Converts all OGLE3 data files in specified directory into light curves
    as lists of the following values: classLabels, times, magnitudes, and
    errors. Class labels are parsed from originating data file name."""
    labels = list()
    times = list()
    magnitudes = list()
    errors = list()
    for i, f in enumerate(absoluteFilePaths(dataDir, ext="dat")):
        if i == limit:
            break

        fileName = f.split("/")[-1]
        fnSplits = fileName.split("-")
        if len(fnSplits) > 2:
            category = fnSplits[2].lower()
        else:
            logger.warning("file name lacks category! %s", fileName)
            continue

        timeSeries, magnitudeSeries, errorSeries = parseOgle3Lc(f)
        labels.append(category)
        times.append(timeSeries)
        magnitudes.append(magnitudeSeries)
        errors.append(errorSeries)

    return labels, times, magnitudes, errors


def reportClassHistogram(labels):
    total = float(len(labels))
    c = Counter([x for x in labels])
    t = PrettyTable(["category", "count", "percentage"])
    t.align = "l"
    for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
        t.add_row([k, v, "{:.2%}".format(v / total)])

    logger.info("\n" + str(t))


def extractFeatures(lcs):
    features = list()
    exclude = [] if False else ["CAR_mean", "CAR_sigma", "CAR_tau"]
    logger.info("Excluded features: %s", exclude)
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)

    startTime = time.time()
    for _, tm, mag, err in lcs:
        _, ftValues = fs.extract(time=tm, magnitude=mag, error=err)
        features.append(ftValues)

    logger.info("extract in: %.02fs", time.time() - startTime)
    return features


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampleLimit", type=int, default=50,
                        help="limit on the number of light curves to process")
    parser.add_argument("-a", "--trainRatio", type=float, default=0.75,
                        help="ratio of desired train set size to entire "
                             "dataset size")

    # TODO support feature
    parser.add_argument("-c", "--cvRatio", type=float, default=0.0,
                        help="ratio of desired cross-validation set size to "
                             "entire dataset length")
    parser.add_argument("--allFeatures", action="store_true",
                        help="specified, all 'feets' features will be "
                             "extracted, otherwise, slow features will be "
                             "omitted")
    return parser.parse_args()


def main():
    """Runs feets on ogle"""
    args = _getArgs()
    start = time.time()

    dataDir = joinRoot("data/ogle3")
    unarchiveAll(dataDir, remove=True)

    _labels, _times, _mags, _errors = ogle3ToLc(dataDir, limit=args.sampleLimit)
    reportClassHistogram(_labels)
    labels, times, mags, errors = cleanDataset(_labels, _times, _mags, _errors,
                                               {float("nan")})
    reportClassHistogram(labels)
    categoryToLabel = convertClassLabels(labels)
    print("Category to label mapping: %s" % categoryToLabel)

    # run data set through feets library obtaining feature vectors
    exclude = [] if args.allFeatures else ["CAR_mean", "CAR_sigma", "CAR_tau"]
    logger.info("Excluded features: %s", exclude)
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    cleanLcDf = [(fs, labels[i], times[i], mags[i], errors[i])
                 for i in range(len(labels))]
    featureLabels, elapsedMin = mapMultiprocess(feetsExtract, cleanLcDf)
    features = [fl[0] for fl in featureLabels]
    classLabels = [fl[1] for fl in featureLabels]

    # create a test and train set
    xTrain, xTest, yTrain, yTest = train_test_split(features, classLabels,
                                                    train_size=args.trainRatio)

    # Train and Test dataset size details
    print("\nTrain & Test sizes")
    print("Train_x Shape: ", len(xTrain))
    print("Train_y Shape: ", len(yTrain))
    print("Test_x Shape: ", len(xTest))
    print("Test_y Shape: ", len(yTest))

    # train RF on train set feature vectors
    model = RandomForestClassifier()
    model.fit(xTrain, yTrain)
    # TODO pickle the RF to disk
    # http://scikit-learn.org/stable/modules/model_persistence.html

    trainPredictions = model.predict(xTrain)
    testPredictions = model.predict(xTest)

    print("Train accuracy: %.5f" % accuracy_score(yTrain, trainPredictions))
    print("Test accuracy: %.5f" % accuracy_score(yTest, testPredictions))
    print("Confusion: \n", confusion_matrix(yTest, testPredictions))

    # TODO
    # research performance metrics from Kim's papers
    # record performance and time to process
    # create CV set and try RF variations
    elapsed = time.time() - start
    print("Completed in: %.3fs" % (elapsed))
    print("time per lc: %.3fs" % (elapsed / len(features)))


if __name__ == "__main__":
    main()
