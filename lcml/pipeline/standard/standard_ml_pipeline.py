import argparse
import time

from prettytable import PrettyTable

from lcml.pipeline.data_format.db_format import connFromParams
from lcml.pipeline.ml_pipeline import fromRelativePath
from lcml.pipeline.model_selection import selectBestModel
from lcml.pipeline.persistence import loadModels, saveModel
from lcml.pipeline.preprocess import cleanLightCurves
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


def reportModelSelection(bestResult, allResults, classToLabel, places):
    """Reports the hyperparameters and associated metrics obtain from model
    selection."""
    reportColumns = ["Hyperparameters", "F1 (micro)", "F1 (class)", "Accuracy"]
    roundFlt = truncatedFloat(places)
    searchTable = PrettyTable(reportColumns)
    for result in allResults:
        searchTable.add_row(_resultToRow(result, classToLabel, roundFlt))

    winnerTable = PrettyTable(reportColumns)
    winnerTable.add_row(_resultToRow(bestResult, classToLabel, roundFlt))

    logger.info("Model search results...\n" + str(searchTable))
    logger.info("Winning model...\n" + str(winnerTable))


def _resultToRow(result, classToLabel, roundFlt):
    """Converts a ModelSelectionResult to a list of formatted values to be used
    as a row in a table"""
    microF1 = roundFlt % result.metrics.f1Overall
    classF1s = [(l, roundFlt % v)
                for l, v
                in attachLabels(result.metrics.f1Individual, classToLabel)]
    accuracy = roundFlt % (100 * result.metrics.accuracy)
    return [result.hyperparameters, microF1, classF1s, accuracy]


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

    pipe.loadData.fcn(loadParams, pipe.dbParams)

    logger.info("Cleaning dataset...")
    cleanLightCurves(loadParams, pipe.dbParams)

    histogram = classLabelHistogram(pipe.dbParams)
    reportClassHistogram(histogram)

    extractParams = pipe.extractFeatures.params
    if not extractParams.get("skip", False):
        extractStart = time.time()
        pipe.extractFeatures.fcn(extractParams, pipe.dbParams)
        extractMins = (time.time() - extractStart) / 60
        logger.info("extracted in %.2fm", extractMins)

    models = None
    loadPath = pipe.serialParams["loadPath"]
    if loadPath:
        # load previous winning model and its metadata from disk
        models = loadModels(loadPath)

    if not models:
        models = pipe.modelSelection.fcn(pipe.modelSelection.params)

    bestResult, allResults, classToLabel = selectBestModel(models,
        pipe.modelSelection.params, pipe.dbParams)
    if pipe.serialParams["savePath"]:
        saveModel(bestResult, pipe.serialParams["savePath"], pipe,
                  classToLabel)

    elapsedMins = (time.time() - startAll) / 60
    logger.info("Pipeline completed in: %.3f min", elapsedMins)
    reportModelSelection(bestResult, allResults, classToLabel,
                         pipe.globalParams.get("places", 3))

    logger.info("Integer class label mapping %s", classToLabel)
    classLabels = [classToLabel[i] for i in sorted(classToLabel)]
    plotConfusionMatrix(bestResult.metrics.confusionMatrix, classLabels,
                        normalize=True)


def classLabelHistogram(dbParams):
    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    histogramQry = "SELECT label, COUNT(*) FROM %s GROUP BY label"
    cursor = cursor.execute(histogramQry % dbParams["clean_lc_table"])
    histogram = dict([_ for _ in cursor])
    conn.close()
    return histogram


if __name__ == "__main__":
    main()
