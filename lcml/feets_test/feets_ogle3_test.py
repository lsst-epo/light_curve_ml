import argparse
from collections import Counter
import os
import time

from feets import FeatureSpace
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from lcml.common import STANDARD_INPUT_DATA_TYPES
from lcml.processing.preprocess import cleanDataset
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.context_util import absoluteFilePaths, ensureRootPath, joinRoot
from lcml.utils.data_util import convertClassLabels, unarchiveAll
from lcml.utils.format_util import fmtPct
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess


logger = getBasicLogger(__name__, __file__)


#: data value to scrub
REMOVE_SET = {float("nan"), float("inf"), float("-inf")}


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sampleLimit", type=int, default=50,
                        help="limit on the number of light curves to process")
    parser.add_argument("-a", "--trainRatio", type=float, default=0.75,
                        help="ratio of desired train set size to entire "
                             "dataset size")

    parser.add_argument("--skipTrain", action="store_true", help="attempt to "
                        "load model from disk instead of training model")
    # TODO support feature
    parser.add_argument("-c", "--cvRatio", type=float, default=0.0,
                        help="ratio of desired cross-validation set size to "
                             "entire dataset length")
    parser.add_argument("--allFeatures", action="store_true",
                        help="if specified, all 'feets' features will be "
                             "extracted, otherwise, slow features will be "
                             "omitted")
    return parser.parse_args()


def loadOgle3Dataset(dataDir, limit=float("inf")):
    """Loads all OGLE3 data files from specified directory as light curves
    represented as lists of the following values: classLabels, times,
    magnitudes, and magnitude errors. Class labels are parsed from originating
    data file name."""
    labels = list()
    times = list()
    magnitudes = list()
    errors = list()
    paths = absoluteFilePaths(dataDir, ext="dat")
    logger.info("Found %s files", len(paths))
    for i, f in enumerate(paths):
        if i == limit:
            break

        fileName = f.split("/")[-1]
        fnSplits = fileName.split("-")
        if len(fnSplits) > 2:
            category = fnSplits[2].lower()
        else:
            logger.warning("file name lacks category! %s", fileName)
            continue

        timeSeries, magnitudeSeries, errorSeries = _parseOgle3Lc(f)
        labels.append(category)
        times.append(timeSeries)
        magnitudes.append(magnitudeSeries)
        errors.append(errorSeries)

    return labels, times, magnitudes, errors


def _parseOgle3Lc(filePath):
    """Parses a light curve as a tuple of numpy arrays from specified OGLE3
    file."""
    lc = np.loadtxt(filePath)
    if lc.ndim == 1:
        lc.shape = 1, 3

    return lc[:, 0], lc[:, 1], lc[:, 2]


def reportClassHistogram(labels):
    """Logs a histogram of the distribution of class labels"""
    c = Counter([l for l in labels])
    t = PrettyTable(["category", "count", "percentage"])
    t.align = "l"
    for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
        t.add_row([k, v, fmtPct(v, len(labels))])

    logger.info("Class histogram:\n" + str(t))


def multiprocessExtract(errors, labels, mags, times, allFeatures):
    # run data set through feets library obtaining feature vectors
    exclude = [] if allFeatures else ["CAR_mean", "CAR_sigma", "CAR_tau"]
    logger.info("Excluded features: %s", exclude)
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    cleanLcDf = [(fs, labels[i], times[i], mags[i], errors[i])
                 for i in range(len(labels))]
    featureLabels, _ = mapMultiprocess(feetsExtract, cleanLcDf)
    validFeatures = []
    validLabels = []
    badCount = 0
    for features, label in featureLabels:
        if allFinite(features):
            validFeatures.append(features)
            validLabels.append(label)
        else:
            # TODO could impute 0.0 or the average value for the feature,
            # if the number of bad values is small, then it's only 1 / 60  and
            # hopefully the rest of the feature vector dominates
            logger.warning("skipping over bad feature set: %s", features)
            badCount += 1

    if badCount:
        logger.warning("Skipped b/c nan rate: %s", fmtPct(badCount,
                                                          len(featureLabels)))

    return validFeatures, validLabels


def allFinite(X):
    """Adapted from sklearn.utils.validation._assert_all_finite"""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    return (False
            if X.dtype.char in np.typecodes['AllFloat'] and
               not np.isfinite(X.sum()) and not np.isfinite(X).all()
            else True)


def main():
    """Runs feets on ogle and classifies resultant features with a RF."""
    args = _getArgs()

    dataDir = joinRoot("data/ogle3")
    unarchiveAll(dataDir, remove=True)
    _labels, _times, _mags, _errors = loadOgle3Dataset(dataDir,
                                                       limit=args.sampleLimit)
    labels, times, mags, errors = cleanDataset(_labels, _times, _mags, _errors,
                                               REMOVE_SET)
    reportClassHistogram(labels)
    intToClassLabel = convertClassLabels(labels)

    startAll = time.time()
    featuresProcessed, labelsProcessed = multiprocessExtract(errors, labels,
                                                             mags, times,
                                                             args.allFeatures)

    xTrain, xTest, yTrain, yTest = train_test_split(featuresProcessed,
                                                    labelsProcessed,
                                                    train_size=args.trainRatio)
    logger.info("Train size: %s Test size: %s", len(xTrain), len(xTest))

    # train RF on train set feature vectors
    model = None
    modelPath = os.path.join(ensureRootPath("models/ogle3"),
                             "rf-classifier.pkl")
    if args.skipTrain:
        try:
            model = joblib.load(modelPath)
        except Exception as e:
            logger.warning("model loading failed %s" % e)

        logger.info("Loaded model from: %s", modelPath)

    if not model:
        model = RandomForestClassifier()
        s = time.time()
        model.fit(xTrain, yTrain)
        logger.info("Trained model in %.2fs", time.time() - s)
        if not args.skipTrain:
            logger.info("Dumped model to: %s", modelPath)
            joblib.dump(model, modelPath)

    # TODO also save metadata: training data, python source code, version of
    # scikit-learn
    # cross validation score obtained on training data,
    # This should make it possible to check that the cross-validation score is
    # in the same range as before.
    # finally the architecture show be the same for dumping and loading

    trainPredictions = model.predict(xTrain)
    testPredictions = model.predict(xTest)

    logger.info("__Metrics__")
    logger.info("Train accuracy: %.5f", accuracy_score(yTrain, trainPredictions))
    logger.info("Test accuracy: %.5f", accuracy_score(yTest, testPredictions))
    _confusionMat = confusion_matrix(yTest, testPredictions)
    logger.info("Confusion matrix:\n%s", _confusionMat)

    # TODO revisit using the label mapping to convert ints to strings
    # confusionMatrix = pd.crosstab(yTest, testPredictions, margins=True)
    # logger.info("\n" + str(confusionMatrix))
    logger.info(["%s=%s" % (i, label)
                 for i, label in sorted(intToClassLabel.items())])

    # TODO
    # research performance metrics from Kim's papers
    # record performance and time to process to file
    # create CV set and try RF variations
    elapsed = time.time() - startAll
    logger.info("Completed in: %.1fs", elapsed)
    logger.info("time per lc: %.3fs", elapsed / len(featuresProcessed))


if __name__ == "__main__":
    main()
