import argparse
import sqlite3
import time

from prettytable import PrettyTable

from lcml.pipeline.ml_pipeline import fromRelativePath
from lcml.pipeline.model_selection import selectBestModel
from lcml.pipeline.persistence import loadModels, saveModel
from lcml.pipeline.preprocess import cleanLightCurves, NON_FINITE_VALUES
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
    parser.add_argument("--path", required=True,
                        help="relative path to pipeline conf")
    return parser.parse_args()


def reportResults(bestResult, allResults, classToLabel, places):
    columns = ["Hyperparams", "Micro F1", "Class F1", "Accuracy"]
    roundFlt = truncatedFloat(places)
    searchTable = PrettyTable(columns)
    for result in allResults:
        searchTable.add_row(_resultToRow(result, classToLabel, roundFlt))

    logger.info("Search results...\n" + str(searchTable))
    winnerTable = PrettyTable(columns)
    winnerTable.add_row(_resultToRow(bestResult, classToLabel, roundFlt))
    logger.info("Winner...\n" + str(winnerTable))

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
    pipe = fromRelativePath(args.path)
    loadParams = pipe.loadData.params
    dataDir = joinRoot(loadParams["relativePath"])
    if loadParams.get("unarchive", False):
        logger.info("Unarchiving files in %s ...", dataDir)
        unarchiveAll(dataDir, remove=True)

    logger.info("Loading dataset with params: {}...".format(loadParams))
    # TODO use schema
    pipe.loadData.fcn(loadParams)

    logger.info("Cleaning dataset...")
    cleanLightCurves(loadParams)

    histogram = classLabelHistogram(loadParams["dbPath"],
                                    loadParams["clean_lc_table"])
    reportClassHistogram(histogram)
    classToLabel = convertClassLabels(list(histogram))
    logger.info(classToLabel)

    extractStart = time.time()
    extractParams = pipe.extractFeatures.params

    # TODO update to use sqlite db!
    labels, times, mags, errors = [None] * 4
    if False:
        features, labels = pipe.extractFeatures.fcn(labels, times, mags,
                                                    errors, extractParams)
        logger.info("Feets float type: %s", type(features[0][0]).__name__)
        extractMins = (time.time() - extractStart) / 60
        logger.info("extracted in %.2fm", extractMins)

        models = None
        loadPath = pipe.serialParams["loadPath"]
        if loadPath:
            # load previous winning model and its metadata from disk
            models = loadModels(loadPath)

        if not models:
            models = pipe.modelSelection.fcn(pipe.modelSelection.params)

        bestResult, allResults = selectBestModel(models, features, labels,
                                                 pipe.modelSelection.params)

        if pipe.serialParams["savePath"]:
            saveModel(bestResult, pipe.serialParams["savePath"], pipe,
                      classToLabel)

        reportResults(bestResult, allResults, classToLabel,
                      pipe.globalParams.get("places", 3))

        elapsedMins = (time.time() - startAll) / 60
        logger.info("Pipeline completed in: %.3f min\n\n", elapsedMins)


def classLabelHistogram(dbPath, table):
    conn = sqlite3.connect(joinRoot(dbPath))
    cursor = conn.cursor()
    res = cursor.execute("SELECT label, count(*) FROM %s GROUP BY label" %
                         table)
    stuff = [_ for _ in res]
    conn.close()
    return dict(stuff)


if __name__ == "__main__":
    main()
