"""This batch pipeline currently assumes a fixed choice of model type, e.g.,
random forest. It performs model selection for the model type using k-fold cross
validation on the training set. Then the winning model is retrained on the full
training set and evaluated on the test set.

See 'Scenario 2 - Train a model and tune (optimize) its hyperparameters' at:
https://sebastianraschka.com/faq/docs/evaluate-a-model.html
"""

from abc import abstractmethod
import argparse
from datetime import timedelta
import time

from sklearn.model_selection import train_test_split

from lcml.pipeline.database.sqlite_db import (classLabelHistogram,
                                              connFromParams,
                                              selectLabelsFeatures)
from lcml.pipeline.stage.preprocess import cleanLightCurves
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.dataset_util import convertClassLabels, reportClassHistogram


logger = BasicLogging.getLogger(__name__)


def pipelineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="relative path to pipeline conf")
    return parser.parse_args()


class BatchPipeline:
    def __init__(self, conf):
        self.conf = conf
        self.globalParams = conf.globalParams
        self.dbParams = conf.dbParams

        self.loadFcn = conf.loadData.fcn
        self.loadParams = conf.loadData.params

        self.extractFcn = conf.extractFeatures.fcn
        self.extractParams = conf.extractFeatures.params

        self.searchFcn = conf.modelSearch.fcn
        self.searchParams = conf.modelSearch.params

        self.serParams = conf.serialParams

    def runPipe(self):
        """Runs initial phase of batch machine learning pipeline storing
        intermediate results in a database. Performs following stages:
        1) parse light curves from data source and store to db
        2) preprocess light curves and store cleaned LC's to db
        3) compute features and store to db
        """
        logger.info("___Begin batch ML pipeline___")
        startAll = time.time()

        if self.loadParams.get("skip", False):
            logger.info("Skip load and clean dataset")
        else:
            logger.info("Loading dataset...")
            self.loadFcn(self.loadParams, self.dbParams)

            logger.info("Cleaning dataset...")
            cleanLightCurves(self.loadParams, self.dbParams)

        logger.info("Cleaned dataset class histogram...")
        histogram = classLabelHistogram(self.dbParams)
        reportClassHistogram(histogram)

        if self.extractParams.get("skip", False):
            logger.info("Skip extract features")
        else:
            logger.info("Extracting features...")
            extractStart = time.time()
            self.extractFcn(self.extractParams, self.dbParams)
            extractMins = (time.time() - extractStart) / 60
            logger.info("extracted in %.2fm", extractMins)

        limit = self.globalParams.get("dataLimit", None)
        conn = connFromParams(self.dbParams)
        cursor = conn.cursor()
        labels, features = selectLabelsFeatures(cursor, self.dbParams, limit)
        conn.close()

        logger.info("Loaded %s feature vectors", len(features))
        intLabels, intToStrLabels = convertClassLabels(labels)

        trainSize = self.globalParams["trainSize"]
        XTrain, XTest, yTrain, yTest = train_test_split(features, intLabels,
                                                        train_size=trainSize,
                                                        test_size=1 - trainSize)
        logger.info("train size: %s test size: %s", len(XTrain), len(XTest))
        modelResult = self.modelSelectionPhase(XTrain, yTrain, intToStrLabels)
        self.evaluateTestSet(modelResult, XTest, yTest, intToStrLabels)

        elapsedMins = timedelta(seconds=(time.time() - startAll))
        logger.info("Pipeline completed in: %s", elapsedMins)

    @abstractmethod
    def modelSelectionPhase(self, trainFeatures, trainLabels, classLabel):
        """Performs model selection on the training set and returns the selected
        model trained on the full training set"""

    @abstractmethod
    def evaluateTestSet(self, model, featuresTest, labelsTest, classLabels):
        """Evaluates specified model on the held-out test set."""
