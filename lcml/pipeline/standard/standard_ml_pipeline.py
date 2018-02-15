import argparse
import time

from lcml.pipeline.ml_pipeline import fromRelativePath
from lcml.pipeline.model_selection import selectBestModel
from lcml.pipeline.persistence import loadModel, saveModel
from lcml.pipeline.preprocess import cleanDataset
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.context_util import joinRoot
from lcml.utils.data_util import (attachLabels, convertClassLabels,
                                  unarchiveAll, reportClassHistogram)
from lcml.utils.format_util import truncatedFloat


logger = getBasicLogger(__name__, __file__)


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelinePath", required=True,
                        help="relative path to pipeline conf")
    return parser.parse_args()


def reportResults(bestResult, allResults, classToLabel, places):
    roundFlt = truncatedFloat(places)
    for result in allResults:
        _reportResult(result, classToLabel, roundFlt)

    logger.info("")
    logger.info("__ Winning model __")
    _reportResult(bestResult, classToLabel, roundFlt)

    # TODO visualization, reporting
    # decide on a function that supports labeling, normalizing, etc
    # confusionMatrix = confusion_matrix(featuresProcessed, predicted)
    # confusionMatrix = pd.crosstab(yTest, testPredictions, margins=True)

    # nice to have - true-class-normalized confusion matrix where cells have
    # float & grayscale intensity

    # nice to have - using confusion matrix, compute, for each
    # class, the precision, recall, f1 with weighted average overall measure


def _reportResult(result, classToLabel, roundFlt):
    logger.info("hyperparams: %s", result.hyperparameters)
    logger.info("micro F1: " + roundFlt, result.metrics.f1Overall)
    labeledF1s = [(l, roundFlt % v)
                  for l, v
                  in attachLabels(result.metrics.f1Individual, classToLabel)]
    logger.info("class F1: %s", labeledF1s)
    logger.info("accuracy: " + roundFlt, result.metrics.accuracy)


def main():
    """Runs a standard machine learning pipeline. With the following stages:
    1) load data
    2) preprocess light curves
    3) compute features
    4) peform model selection using k-fold cross validation
    5) serialize model to disk
    6) report metrics
    """
    startAll = time.time()
    args = _getArgs()
    pipe = fromRelativePath(args.pipelinePath)
    loadParams = pipe.loadData.params
    dataDir = joinRoot(loadParams["relativePath"])
    if loadParams["unarchive"]:
        logger.info("Unarchiving files in %s ...", dataDir)
        unarchiveAll(dataDir, remove=True)

    logger.info("Loading dataset...")
    _labels, _times, _mags, _errors = pipe.loadData.fcn(dataDir,
                                                        loadParams["limit"])

    logger.info("Cleaning dataset...")
    labels, times, mags, errors = cleanDataset(_labels, _times, _mags, _errors)
    reportClassHistogram(labels)
    classToLabel = convertClassLabels(labels)

    featuresStart = time.time()
    extractParams = pipe.extractFeatures.params
    features, labelsProcessed = pipe.extractFeatures.fcn(labels, times, mags,
                                                         errors, extractParams)
    logger.info("extracted in %.2fs", time.time() - featuresStart)

    models = None
    if pipe.serialParams["modelPath"]:
        # load model and its metadata from disk
        _model, _metadata = loadModel(pipe.serialParams["modelPath"])
        if _model and _metadata:
            models = [(_model, _metadata["hyperparameters"])]

    if not models:
        models = pipe.modelSelection.fcn(pipe.modelSelection.params)

    bestResult, allResults = selectBestModel(models, features, labelsProcessed,
                                             pipe.modelSelection.params)
    logger.info("Finished searching")

    allParams = {"hyperparameters": bestResult.hyperparameters,
                 "loadParams": loadParams, "extractParams": extractParams,
                 "selectionParams": pipe.modelSelection.params}
    if pipe.serialParams["saveModel"]:
        saveModel(bestResult.model, pipe.serialParams["modelPath"], allParams,
                  bestResult.metrics)

    reportResults(bestResult, allResults, classToLabel,
                  pipe.globalParams["places"])

    elapsedMins = (time.time() - startAll) / 60
    logger.info("Completed in: %.3f min", elapsedMins)


if __name__ == "__main__":
    main()
