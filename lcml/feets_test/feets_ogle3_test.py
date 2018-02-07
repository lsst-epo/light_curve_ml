import argparse
from collections import Counter
import json
import os
import platform
import time

from feets import FeatureSpace
import numpy as np
from prettytable import PrettyTable
from scipy import stats
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     cross_validate, train_test_split)

from lcml.common import STANDARD_INPUT_DATA_TYPES
from lcml.processing.preprocess import cleanDataset
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.context_util import absoluteFilePaths, joinRoot
from lcml.utils.data_util import convertClassLabels, unarchiveAll
from lcml.utils.format_util import fmtPct
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess


logger = getBasicLogger(__name__, __file__)


#: data value to scrub
REMOVE_SET = {float("nan"), float("inf"), float("-inf")}


#: N.B. common classification scoring types from:
#: scikit-learn.org/stable/modules/model_evaluation.html
DEFAULT_SCORING = ["accuracy", "average_precision", "f1", "f1_micro",
                   "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
                   "precision", "recall", "roc_auc"]


_REL_DATA_DIR = "data/ogle3"


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--sampleLimit", type=int, default=50,
                        help="limit on the number of light curves to process")
    parser.add_argument("-c", "--cv", type=int, default=5,
                        help="number of cross-validation folds")
    parser.add_argument("--allFeatures", action="store_true",
                        help="if specified, all 'feets' features will be "
                             "extracted, otherwise, slow features will be "
                             "omitted")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="number of processes to use for ML model "
                             "computation. -1 implies all available cores")
    parser.add_argument("-l", "--modelPath", type=str, help="path from "
                        "which to load random forest classifier")
    parser.add_argument("-s", "--saveModel", action="store_true",
                        help="Specify to save the model to the 'loadModelPath'")

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
    for i, f in enumerate(paths[:limit]):
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
    logger.info("Trained model in %.2fs", time.time() - s)
    return model


def saveModel(model, modelPath, hyperparams=None, metrics=None):
    """If 'modelPath' is specified, the model and its metadata, including
    'trainParams' and 'cvScore' are saved to disk.

    :param model: a trained ML model, could be any Python object
    :param modelPath: save path
    :param hyperparams: all model hyperparameters
    :param metrics: metric values obtained from running model on test data
    """
    joblib.dump(model, modelPath)
    logger.info("Dumped model to: %s", modelPath)
    metadataPath = _metadataPath(modelPath)
    archBits = platform.architecture()[0]
    metadata = {"archBits": archBits, "sklearnVersion": sklearn.__version__,
                "pythonSource": __name__, "hyperparameters": hyperparams,
                "metrics": metrics}
    with open(metadataPath, "w") as f:
        json.dump(metadata, f)


def loadModel(modelPath):
    try:
        model = joblib.load(modelPath)
    except IOError:
        logger.warning("Failed to load model from: %s", modelPath)
        return None

    logger.info("Loaded model from: %s", modelPath)
    metadataPath = _metadataPath(modelPath)
    try:
        with open(metadataPath) as mFile:
            metadata = json.load(mFile)
    except IOError:
        logger.warning("Metadata file doesn't exist: %s", metadataPath)
        return model

    if metadata["archBits"] != platform.architecture()[0]:
        logger.critical("Model created on arch: %s but current arch is %s",
                        metadata["archBits"], platform.architecture()[0])
        raise ValueError("Unusable model")

    logger.info("Model metadata: %s", metadata)
    return model


def _metadataPath(modelPath):
    finalDirLoc = modelPath.rfind(os.sep)
    return os.path.join(modelPath[:finalDirLoc], "metadata.json")


def main():
    """Runs feets on ogle and classifies resultant features with a RF."""
    startAll = time.time()
    args = _getArgs()
    dataDir = joinRoot(_REL_DATA_DIR)
    unarchiveAll(dataDir, remove=True)
    _labels, _times, _mags, _errors = loadOgle3Dataset(dataDir,
                                                       limit=args.sampleLimit)
    labels, times, mags, errors = cleanDataset(_labels, _times, _mags, _errors,
                                               REMOVE_SET)
    reportClassHistogram(labels)
    intToClassLabel = convertClassLabels(labels)

    logger.info("Extracting features...")
    featuresStart = time.time()
    featuresProcessed, labelsProcessed = multiprocessExtract(errors, labels,
                                                             mags, times,
                                                             args.allFeatures)
    logger.info("took %.2fs", time.time() - featuresStart)

    models = None
    if args.modelPath:
        # consider a separate script for just running a serialized model
        # try request to load model from disk
        models = [(0, 0, loadModel(args.modelPath))]

    if not models:
        # default for num estimators is 10
        estimatorsStart = 5
        estimatorsStop = 16

        # default for max features is sqrt(len(features))
        # for feets len(features) ~= 64 => 8
        rfFeaturesStart = 5
        rfFeaturesStop = 11
        models = [(t, f, RandomForestClassifier(n_estimators=t, max_features=f,
                                                n_jobs=args.jobs))
                  for f in range(rfFeaturesStart, rfFeaturesStop)
                  for t in range(estimatorsStart, estimatorsStop)]

    scoring = ["accuracy"]
    bestModel, bestParams, bestMetrics = searchBestModel(models,
                                                         featuresProcessed,
                                                         labelsProcessed,
                                                         scoring,
                                                         args.cv, args.jobs)
    logger.info("")
    logger.info("__ Winning model __")
    logger.info("hyperparameters: %s", bestParams)
    logger.info("accuracy: %.5fs", bestMetrics["test_accuracy"])

    bestParams["allFeatures"] = args.allFeatures
    bestParams["cv"] = args.cv
    if args.saveModel:
        # save regardless of args.modelPath value
        saveModel(bestModel, args.modelPath, bestParams, bestMetrics)

    logger.info("Label mapping: %s", ["%s=%s" % (i, label)
                                      for i, label in
                                      sorted(intToClassLabel.items())])
    elapsedMins = (time.time() - startAll) / 60
    logger.info("Completed in: %.3f min", elapsedMins)


def searchBestModel(models, features, labels, scoring, cv, jobs):
    """Tries all specified models and select one with highest F1 score". Returns
    the best model, the model's associated hyperparameters, """
    bestModel = None
    bestTrees = None
    bestMaxFeats = None
    bestMetrics = None
    maxAveAccuracy = 0
    fitTimes = []
    scoreTimes = []
    for trees, maxFeats, model in models:
        logger.info("trees: %s max features: %s", trees, maxFeats)
        scores = cross_validate(model, features, labels, scoring=scoring, cv=cv,
                                n_jobs=jobs, return_train_score=False)

        scores["test_accuracy_mean"] = np.mean(scores["test_accuracy"])
        scores["test_accuracy_ci"] = confidenceInterval(scores["test_accuracy"],
            scores["test_accuracy_mean"])

        fitTimes.append(np.average(scores.pop("fit_time")))
        scoreTimes.append(np.average(scores.pop("score_time")))
        if scores["test_accuracy_mean"] > maxAveAccuracy:
            maxAveAccuracy = scores["test_accuracy_mean"]
            bestModel = model
            bestTrees = trees
            bestMaxFeats = maxFeats
            bestMetrics = scores


        logger.info("accuracy: %.7f ci: %.7f %.7f",
                    scores["test_accuracy_mean"],
                    scores["test_accuracy_ci"][0],
                    scores["test_accuracy_ci"][1])

        # trying other route
        predicted = cross_val_predict(model, features, labels, cv=cv,
                                      n_jobs=jobs)
        logger.info("current accuracy approach: %.5f new approach %.5f",
                    scores["test_accuracy_mean"],
                    accuracy_score(labels, predicted))

        # TODO true class normalized confusion matrix with number and grayscale
        # intensity
        # replace above with:
        # predicted = cross_val_predict(model, features, labels, cv=cv,
        #                               n_jobs=jobs)
        # _confusionMat = confusion_matrix(featuresProcessed, predicted)
        # logger.info("Confusion matrix:\n%s", _confusionMat)
        # TO DO revisit using the label mapping to convert ints to strings
        # confusionMatrix = pd.crosstab(yTest, testPredictions, margins=True)
        # logger.info("\n" + str(confusionMatrix))

        # TODO using confusion matrix, compute, for each class, the precision,
        # recall, f1 with weighted average overall measure

    logger.info("average fit time: %.2fs", np.average(fitTimes))
    logger.info("average score time: %.2fs", np.average(scoreTimes))
    return (bestModel, {"trees": bestTrees, "maxFeatures": bestMaxFeats},
            bestMetrics)


def confidenceInterval(values, mean, confidence=0.99):
    """Calculates confidence interval using Student's t-distribution and
    standard error of mean"""
    return stats.t.interval(confidence, len(values) - 1, loc=mean,
                            scale=stats.sem(values))


if __name__ == "__main__":
    main()
