import argparse
import time

from feets import FeatureSpace
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_validate

from lcml.pipeline.data_format import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.ml_pipeline import fromRelativePath
from lcml.pipeline.persistence import loadModel, saveModel
from lcml.pipeline.preprocess import cleanDataset, allFinite
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.context_util import joinRoot
from lcml.utils.data_util import (attachLabels, convertClassLabels,
                                  unarchiveAll, reportClassHistogram)
from lcml.utils.format_util import fmtPct, truncatedFloat
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess
from lcml.utils.stats_utils import confidenceInterval


logger = getBasicLogger(__name__, __file__)


#: N.B. common classification scoring types from:
#: scikit-learn.org/stable/modules/model_evaluation.html
DEFAULT_SCORING = ["accuracy", "average_precision", "f1", "f1_micro",
                   "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
                   "precision", "recall", "roc_auc"]


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelinePath", required=True,
                        help="relative path to pipeline conf")
    parser.add_argument("-u", "--unarchive", action="store_true",
                        help="unarchive files in data dir")
    parser.add_argument("-t", "--limit", type=int, default=50,
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
    parser.add_argument("-p", "--places", default=5, type=int,
                        help="number digits after the decimal to display")
    return parser.parse_args()


def feetsExtractFeatures(labels, times, mags, errors, exclude=None):
    """Runs light curves through 'feets' library obtaining feature vectors.
    Perfoms the extraction using multiprocessing. Output order will not
    necessarily correspond to input order, therefore, class labels are returned
    as well aligned with feature vectors to avoid confusion.

    :param labels: light curve class labels
    :param times: light curve times
    :param mags: light curve magnitudes
    :param errors: light curve magnitude errors
    :param exclude: features to exclude from computation
    :returns feature vectors for each LC and list of corresponding class labels
    """
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


# TODO this function is doing too much Remove classToLabel, places, scoring?
def searchBestRfModel(models, features, labels, scoring, classToLabel, cv, jobs,
                      places, keyMetric="test_f1_micro"):
    """Tries all specified models and select one with highest key metric score.
    Returns the best model, its hyperparameters, and its scoring metrics."""
    bestModel = None
    bestTrees = None
    bestMaxFeats = None
    bestMetrics = None
    maxMetricValue = 0
    fitTimes = []
    scoreTimes = []
    roundFlt = truncatedFloat(places)
    for trees, maxFeats, model in models:
        logger.info("")
        logger.info("Evaluating: trees: %s max features: %s", trees, maxFeats)

        # TODO remove this and use second approach below to derive accuracy w/o
        # mean and ci
        scores = cross_validate(model, features, labels, scoring=scoring, cv=cv,
                                n_jobs=jobs, return_train_score=False)
        scores["test_accuracy_mean"] = np.mean(scores["test_accuracy"])
        scores["test_accuracy_ci"] = confidenceInterval(scores["test_accuracy"],
            scores["test_accuracy_mean"])

        fitTimes.append(np.average(scores.pop("fit_time")))
        scoreTimes.append(np.average(scores.pop("score_time")))

        # this method allows us more flexibility in computing metrics
        predicted = cross_val_predict(model, features, labels, cv=cv,
                                      n_jobs=jobs)

        origAccu = scores["test_accuracy_mean"]
        newAccu = accuracy_score(labels, predicted)

        # TODO remove
        accuPercentDiff = fmtPct(abs(origAccu - newAccu), origAccu, places)
        logger.info("percent diff in accuracy: %s", accuPercentDiff)

        scores["test_f1_micro"] = f1_score(labels, predicted, average="micro")
        scores["test_f1_class"] = [round(x, places) for x in
                                   f1_score(labels, predicted, average=None)]

        # TODO instead of logging here, simple return the scores in a list
        logger.info("micro F1: " + roundFlt, scores["test_f1_micro"])
        logger.info("class F1: %s", attachLabels(scores["test_f1_class"],
                                                 classToLabel))
        logger.info("accuracy: %s ci: %s %s" % (roundFlt,  roundFlt,  roundFlt),
                    scores["test_accuracy_mean"],
                    scores["test_accuracy_ci"][0],
                    scores["test_accuracy_ci"][1])
        if scores[keyMetric] > maxMetricValue:
            maxMetricValue = scores[keyMetric]
            bestModel = model
            bestTrees = trees
            bestMaxFeats = maxFeats
            bestMetrics = scores

    logger.info("")
    logger.info("Finished searching")
    logger.info("average fit time: %.2fs", np.average(fitTimes))
    logger.info("average score time: %.2fs", np.average(scoreTimes))
    return (bestModel, {"trees": bestTrees, "maxFeatures": bestMaxFeats},
            bestMetrics)


def reportResults(bestParams, bestMetrics, classToLabel, places):
    logger.info("")
    logger.info("__ Winning model __")
    logger.info("hyperparameters: %s", bestParams)
    logger.info("mean accuracy: " + truncatedFloat(places),
                bestMetrics["test_accuracy_mean"])
    logger.info("micro F1: " + truncatedFloat(places),
                bestMetrics["test_f1_micro"])
    logger.info("class F1: %s", attachLabels(bestMetrics["test_f1_class"],
                                             classToLabel))

    # TODO visualization, reporting
    # decide on a function that supports labeling, normalizing, etc
    # confusionMatrix = confusion_matrix(featuresProcessed, predicted)
    # confusionMatrix = pd.crosstab(yTest, testPredictions, margins=True)

    # nice to have - true-class-normalized confusion matrix where cells have
    # float & grayscale intensity

    # nice to have - using confusion matrix, compute, for each
    # class, the precision, recall, f1 with weighted average overall measure


def gridSearchSelection(params):
    # default for num estimators is 10
    estimatorsStart = params["estimatorsStart"]
    estimatorsStop = params["estimatorsStop"]

    # default for max features is sqrt(len(features))
    # for feets len(features) ~= 64 => 8
    rfFeaturesStart = params["rfFeaturesStart"]
    rfFeaturesStop = params["rfFeaturesStop"]
    return [(t, f, RandomForestClassifier(n_estimators=t, max_features=f,
                                          n_jobs=params["jobs"]))
            for f in range(rfFeaturesStart, rfFeaturesStop)
            for t in range(estimatorsStart, estimatorsStop)]


def main():
    """Runs feets on ogle and classifies resultant features with a RF."""
    startAll = time.time()
    args = _getArgs()
    pipe = fromRelativePath(args.pipelinePath)

    _REL_DATA_DIR = "../data/ogle3"  # TODO add to conf
    dataDir = joinRoot(_REL_DATA_DIR)

    if args.unarchive:
        logger.info("Unarchiving files in %s ...", dataDir)
        unarchiveAll(dataDir, remove=True)

    logger.info("Loading dataset...")
    _labels, _times, _mags, _errors = pipe.loadData.fcn(dataDir, args.limit)

    logger.info("Cleaning dataset...")
    labels, times, mags, errors = cleanDataset(_labels, _times, _mags, _errors)
    reportClassHistogram(labels)
    classToLabel = convertClassLabels(labels)

    featuresStart = time.time()
    exclude = [] if args.allFeatures else ["CAR_mean", "CAR_sigma", "CAR_tau"]
    logger.info("Excluded features: %s", exclude)
    logger.info("Extracting features...")
    featuresProcessed, labelsProcessed = feetsExtractFeatures(labels, times,
                                                              mags, errors,
                                                              exclude)
    logger.info("extracted in %.2fs", time.time() - featuresStart)

    models = None
    if args.modelPath:
        # consider a separate script for just running a serialized model
        # try request to load model from disk
        models = [(0, 0, loadModel(args.modelPath))]

    if not models:
        # TODO fcn from pipe
        models = gridSearchSelection(pipe.modelSelection.args)

    scoring = ["accuracy"]
    bestModel, bestParams, bestMetrics = searchBestRfModel(models,
                                                           featuresProcessed,
                                                           labelsProcessed,
                                                           scoring,
                                                           classToLabel,
                                                           args.cv, args.jobs,
                                                           args.places)
    bestParams["allFeatures"] = args.allFeatures
    bestParams["cv"] = args.cv
    if args.saveModel:
        saveModel(bestModel, args.modelPath, bestParams, bestMetrics)

    reportResults(bestParams, bestMetrics, classToLabel, args.places)
    elapsedMins = (time.time() - startAll) / 60
    logger.info("Completed in: %.3f min", elapsedMins)


if __name__ == "__main__":
    main()
