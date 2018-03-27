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

        self.selectionFcn = conf.modelSelection.fcn
        self.selectionParams = conf.modelSelection.params

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


        conn = connFromParams(self.dbParams)
        cursor = conn.cursor()
        limit = self.globalParams.get("dataLimit", None)
        labels, features = selectLabelsFeatures(cursor, self.dbParams, limit)
        conn.close()

        logger.info("Loaded %s feature vectors", len(features))
        classLabels = convertClassLabels(labels)

        trainSize = self.globalParams["trainSize"]
        featTrain, labelsTrain, featTest, labelsTest = train_test_split(
            features, labels, train_size=trainSize)
        logger.info("train set size: %s test set size: %s", len(labelsTrain),
                    len(labelsTest))
        winner = self.modelSelectionPhase(featTrain, labelsTrain, classLabels)
        self.evaluateTestSet(winner, featTest, labelsTest, classLabels)

        elapsedMins = timedelta(seconds=(time.time() - startAll))
        logger.info("Pipeline completed in: %s", elapsedMins)

    @abstractmethod
    def modelSelectionPhase(self, trainFeatures, trainLabels, classLabel):
        """Evaluate ML models, generate performance measures, report results"""

    @abstractmethod
    def evaluateTestSet(self, model, featuresTest, labelsTest, classLabels):
        """ TODO """
