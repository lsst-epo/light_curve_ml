import argparse
import time

from prettytable import PrettyTable

from lcml.pipeline.ml_pipeline import fromRelativePath
from lcml.pipeline.model_selection import selectBestModel
from lcml.pipeline.persistence import loadModel, saveModel
from lcml.pipeline.preprocess import cleanDataset
from lcml.pipeline.visualization import plotConfusionMatrix
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import joinRoot
from lcml.utils.data_util import (attachLabels, convertClassLabels,
                                  reportClassHistogram, unarchiveAll)
from lcml.utils.format_util import truncatedFloat


BasicLogging.initLogging()
logger = BasicLogging.getLogger(__name__)


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelinePath", required=True,
                        help="relative path to pipeline conf")
    return parser.parse_args()


def reportResults(bestResult, allResults, classToLabel, places):
    roundFlt = truncatedFloat(places)
    t = PrettyTable(["Hyperparams", "Micro F1", "Class F1", "Accuracy"])
    for result in allResults:
        t.add_row(_resultToRow(result, classToLabel, roundFlt))

    t.add_row(["Winner"] * len(t.field_names))
    t.add_row(_resultToRow(bestResult, classToLabel, roundFlt))
    logger.info("Results...\n" + str(t))

    confusionMatrix = bestResult.metrics.confusionMatrix
    classes = [classToLabel[i] for i in range(len(classToLabel))]
    plotConfusionMatrix(confusionMatrix, classes, normalize=True)


def _resultToRow(result, classToLabel, roundFlt):
    microF1 = roundFlt % result.metrics.f1Overall
    labeledF1s = [(l, roundFlt % v)
                  for l, v
                  in attachLabels(result.metrics.f1Individual, classToLabel)]
    accuracy = roundFlt % (100 * result.metrics.accuracy)
    return [result.hyperparameters, microF1, labeledF1s, accuracy]


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
    loadPath = pipe.serialParams["loadPath"]
    if loadPath:
        # load model and its metadata from disk
        _model, _metadata = loadModel(loadPath)
        if _model is not None and _metadata is not None:
            _hyperparams = _metadata["params"]["hyperparameters"]
            models = [(_model, _hyperparams)]

    if not models:
        models = pipe.modelSelection.fcn(pipe.modelSelection.params)

    bestResult, allResults = selectBestModel(models, features, labelsProcessed,
                                             pipe.modelSelection.params)

    allParams = {"hyperparameters": bestResult.hyperparameters,
                 "loadParams": loadParams, "extractParams": extractParams,
                 "selectionParams": pipe.modelSelection.params}
    if pipe.serialParams["savePath"]:
        metrics = bestResult.metrics._asdict()
        metrics["mapping"] = classToLabel
        saveModel(bestResult.model, pipe.serialParams["savePath"], allParams,
                  metrics)

    reportResults(bestResult, allResults, classToLabel,
                  pipe.globalParams.get("places", 3))

    elapsedMins = (time.time() - startAll) / 60
    logger.info("Pipeline completed in: %.3f min\n\n", elapsedMins)


if __name__ == "__main__":
    main()
