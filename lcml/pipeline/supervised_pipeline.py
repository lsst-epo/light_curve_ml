import numpy as np
import os

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.stage.model_selection import (getClassificationMetrics,
                                                 reportModelSelection,
                                                 selectBestModel)
from lcml.pipeline.stage.persistence import loadModels, saveModel
from lcml.pipeline.stage.visualization import contourPlot, plotConfusionMatrix
from lcml.utils.basic_logging import BasicLogging

logger = BasicLogging.getLogger(__name__)


class SupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self, XTrain, yTrain, intToStrLabel):
        """Runs the supervised portion of a batch machine learning pipeline.
        Performs following stages:
        4) obtain models and peform model selection
        5) serialize winning model to disk
        6) report metrics
        """
        modelHyperparams = None
        modelLoadPath = self.serParams["modelLoadPath"]
        if modelLoadPath:
            modelHyperparams = loadModels(modelLoadPath)
        if not modelHyperparams:
            modelHyperparams = self.searchFcn(self.searchParams)

        model = self.searchParams["model"]
        folds = self.searchParams["folds"]
        repeats = self.searchParams["repeats"]
        bestResult, allResults = selectBestModel(model, modelHyperparams,
                                                 XTrain, yTrain, folds, repeats)
        if self.serParams["modelSavePath"]:
            saveModel(bestResult, self.serParams["modelSavePath"],
                      self.conf, intToStrLabel)

        roundPlaces = self.globalParams["places"]
        reportModelSelection(allResults, intToStrLabel, roundPlaces,
                             title="CV search results")
        reportModelSelection([bestResult], intToStrLabel, roundPlaces,
                             title="Best result")

        imgPath = self.serParams["imgPath"]
        self.plotHyperparamSearch(allResults, imgPath)

        logger.info("Integer class label mapping %s", intToStrLabel)
        classLabels = [intToStrLabel[i] for i in sorted(intToStrLabel)]
        matSavePath = os.path.join(imgPath, "train-set-confusion-matrix.png")
        plotConfusionMatrix(bestResult.metrics.confusionMatrix, classLabels,
                            matSavePath, title="Best-model CV confusion matrix")
        return bestResult

    @staticmethod
    def plotHyperparamSearch(allResults, imgPath):
        # plot effects of hyperparameters on weight-average F1
        x, y, z = zip(*[(r.hyperparameters["n_estimators"],
                         r.hyperparameters["max_features"],
                         r.metrics.f1Weighted)
                        for r in allResults])
        savePath = os.path.join(imgPath, "hyper.png")
        xAxis = sorted(np.unique(x))
        yAxis = sorted(np.unique(y))
        if len(xAxis) > 1 and len(yAxis) > 1:
            title = "RF Hyperparams"
            if np.any([not x or isinstance(x, str) for x in yAxis]):
                title += " - features: " + str(yAxis)
                yAxis = np.arange(len(yAxis))

            zMat = np.array(z).reshape(len(yAxis), len(xAxis))
            contourPlot(xAxis, yAxis, zMat, savePath, title=title,
                        yLabel="trees")

    def evaluateTestSet(self, modelResult, XTest, yTest, intToStrLabels):
        logger.info("Evaluating model on test set...")
        yHat = modelResult.model.predict(XTest)
        metrics = getClassificationMetrics(yTest, yHat)

        imgPath = self.serParams["imgPath"]
        matSavePath = os.path.join(imgPath, "test-set-confusion-matrix.png")
        plotConfusionMatrix(metrics.confusionMatrix, intToStrLabels,
                            matSavePath, title="Test-set confusion matrix")

        reportModelSelection([modelResult], intToStrLabels,
                             title="Test set performance")
