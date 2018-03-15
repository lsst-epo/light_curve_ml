import argparse
import time

from lcml.pipeline.database.sqlite_db import connFromParams
from lcml.pipeline.ml_pipeline import fromRelativePath
from lcml.pipeline.model_selection import reportModelSelection, selectBestModel
from lcml.pipeline.persistence import loadModels, saveModel
from lcml.pipeline.preprocess import cleanLightCurves
from lcml.pipeline.visualization import contourPlot, plotConfusionMatrix
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.dataset_util import reportClassHistogram


BasicLogging.initLogging()
logger = BasicLogging.getLogger(__name__)


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="relative path to pipeline conf")
    return parser.parse_args()


def classLabelHistogram(dbParams):
    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    histogramQry = "SELECT label, COUNT(*) FROM %s GROUP BY label"
    cursor = cursor.execute(histogramQry % dbParams["clean_lc_table"])
    histogram = dict([_ for _ in cursor])
    conn.close()
    return histogram


def main():
    """Runs a batch machine learning pipeline storing intermediate results in a
    database. Consists of the following main stages:
    1) parse light curves from data source and store to db
    2) preprocess light curves and store cleaned LC's to db
    3) compute features and store to db
    4) obatin models and peform model selection
    5) serialize winning model to disk
    6) report metrics
    """
    logger.info("___Begin batch ML pipeline___")
    startAll = time.time()
    args = _getArgs()
    pipe = fromRelativePath(args.path)
    loadParams = pipe.loadData.params

    if loadParams.get("skip", False):
        logger.info("Skip load and clean dataset")
    else:
        logger.info("Loading dataset...")
        pipe.loadData.fcn(loadParams, pipe.dbParams)

        logger.info("Cleaning dataset...")
        cleanLightCurves(loadParams, pipe.dbParams)

    logger.info("Cleaned dataset class histogram...")
    histogram = classLabelHistogram(pipe.dbParams)
    reportClassHistogram(histogram)

    extractParams = pipe.extractFeatures.params
    if extractParams.get("skip", False):
        logger.info("Skip extract features")
    else:
        logger.info("Extracting features...")
        extractStart = time.time()
        pipe.extractFeatures.fcn(extractParams, pipe.dbParams)
        extractMins = (time.time() - extractStart) / 60
        logger.info("extracted in %.2fm", extractMins)

    models = None
    modelLoadPath = pipe.serialParams["modelLoadPath"]
    if modelLoadPath:
        models = loadModels(modelLoadPath)
    if not models:
        models = pipe.modelSelection.fcn(pipe.modelSelection.params)
    bestResult, allResults, classToLabel = selectBestModel(models,
        pipe.modelSelection.params, pipe.dbParams)
    if pipe.serialParams["modelSavePath"]:
        saveModel(bestResult, pipe.serialParams["modelSavePath"], pipe,
                  classToLabel)

    elapsedMins = (time.time() - startAll) / 60
    logger.info("Pipeline completed in: %.3f min", elapsedMins)
    reportModelSelection(bestResult, allResults, classToLabel,
                         pipe.globalParams.get("places", 3))

    # plot effects of hyperparameters on F1-micro
    x, y, z = zip(*[(r.hyperparameters["trees"],
                     r.hyperparameters["maxFeatures"],
                     r.metrics.f1Overall)
                    for r in allResults])
    contourPlot(x, y, z)

    logger.info("Integer class label mapping %s", classToLabel)
    classLabels = [classToLabel[i] for i in sorted(classToLabel)]
    plotConfusionMatrix(bestResult.metrics.confusionMatrix, classLabels,
                        pipe.serialParams["confusionMatrixSavePath"])


if __name__ == "__main__":
    main()
