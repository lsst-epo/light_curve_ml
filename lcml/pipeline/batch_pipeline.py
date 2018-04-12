"""This batch pipeline currently assumes a fixed choice of model type, e.g.,
random forest. It performs model selection for the model type using k-fold cross
validation on the training set. Then the winning model is retrained on the full
training set and evaluated on the test set.

See 'Scenario 2 - Train a model and tune (optimize) its hyperparameters' at:
https://sebastianraschka.com/faq/docs/evaluate-a-model.html
"""
from abc import abstractmethod
from datetime import timedelta
import time

from sklearn.model_selection import train_test_split

from lcml.pipeline.database.sqlite_db import (classLabelHistogram,
                                              ensureDbTables,
                                              selectFeaturesLabels)
from lcml.pipeline.stage.model_selection import (ClassificationMetrics,
                                                 ModelSelectionResult)
from lcml.pipeline.stage.persistence import savePipelineResults
from lcml.pipeline.stage.preprocess import cleanLightCurves
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.dataset_util import convertClassLabels, reportClassHistogram


logger = BasicLogging.getLogger(__name__)


class BatchPipeline:
    def __init__(self, conf):
        self.conf = conf
        self.globalParams = conf.globalParams

        # database params
        self.dbParams = conf.dbParams

        # loading data from external source and converting to valid light curves
        self.loadFcn = conf.loadData.fcn
        self.loadParams = conf.loadData.params

        # extracting feature vectors from light curves
        self.extractFcn = conf.extractFeatures.fcn
        self.extractParams = conf.extractFeatures.params

        # searching over hyperparameters for best model configuration
        self.searchFcn = conf.modelSearch.fcn
        self.searchParams = conf.modelSearch.params

        # serialization of state to disk
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

        ensureDbTables(self.dbParams)

        dataLimit = self.globalParams.get("dataLimit", float("inf"))
        if self.loadParams.get("skip", False):
            logger.info("Skip dataset loading")
        else:
            logger.info("Loading dataset...")
            self.loadFcn(self.loadParams, self.dbParams, dataLimit)

        if self.loadParams.get("skipCleaning", False):
            logger.info("Skip dataset cleaning")
        else:
            logger.info("Cleaning dataset...")
            cleanLightCurves(self.loadParams, self.dbParams, dataLimit)

        logger.info("Cleaned dataset class histogram...")
        histogram = classLabelHistogram(self.dbParams)
        reportClassHistogram(histogram)
        if self.extractParams.get("skip", False):
            logger.info("Skip extract features")
        else:
            logger.info("Extracting features from LCs...")
            extractStart = time.time()
            self.extractFcn(self.extractParams, self.dbParams, dataLimit)
            extractElapsed = timedelta(seconds=time.time() - extractStart)
            logger.info("extracted in %s", extractElapsed)

        features, labels = selectFeaturesLabels(self.dbParams, dataLimit)
        if features:
            logger.info("Loaded %s feature vectors", len(features))
            intLabels, labelMapping = convertClassLabels(labels)
            trainSize = self.globalParams["trainSize"]
            XTrain, XTest, yTrain, yTest = train_test_split(features, intLabels,
                train_size=trainSize, test_size=1 - trainSize)
            logger.info("train size: %s test size: %s", len(XTrain), len(XTest))
            best = self.modelSelectionPhase(XTrain, yTrain, labelMapping)
            testMetrics = self.evaluateTestSet(best, XTest, yTest, labelMapping)
            savePipelineResults(self.conf, labelMapping, best, testMetrics)

        elapsedMins = timedelta(seconds=time.time() - startAll)
        logger.info("Pipeline completed in: %s", elapsedMins)

    @abstractmethod
    def modelSelectionPhase(self, trainFeatures, trainLabels,
                            classLabel) -> ModelSelectionResult:
        """Performs model selection on the training set and returns the selected
        model trained on the full training set"""

    @abstractmethod
    def evaluateTestSet(self, model, featuresTest, labelsTest,
                        classLabels) -> ClassificationMetrics:
        """Evaluates specified model on the held-out test set."""
