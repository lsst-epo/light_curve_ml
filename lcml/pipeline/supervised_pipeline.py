from datetime import timedelta
import numpy as np
import os
import time

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.stage.model_selection import (ClassificationMetrics,
                                                 defaultClassificationMetrics,
                                                 ModelSelectionResult,
                                                 reportModelSelection)
from lcml.pipeline.stage.persistence import loadModelAndHyperparms
from lcml.pipeline.stage.visualization import contourPlot, plotConfusionMatrix
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


class SupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self, XTrain, yTrain,
                            intToStrLabel) -> ModelSelectionResult:
        """Runs the supervised portion of a batch machine learning pipeline.
        Loads a model and its hyperparams from disk if specified or performs
        model selection and to obtain a ModelSelectionResult.
        """
        modelLoadPath = self.serParams["modelLoadPath"]
        if modelLoadPath:
            model, hyperparams = loadModelAndHyperparms(modelLoadPath)
            result = ModelSelectionResult(model, hyperparams, None)
        else:
            start = time.time()
            result = self.searchFcn(self.searchParams["model"], XTrain, yTrain,
                                    self.searchParams["cv"],
                                    self.searchParams["gridSearch"])
            logger.info("search completed in: %s",
                        timedelta(seconds=time.time() - start))

            roundPlaces = self.globalParams["places"]
            reportModelSelection([result.hyperparameters], [result.metrics],
                                 intToStrLabel, roundPlaces,
                                 title="Best result")

        imgPath = self.serParams["imgPath"]
        classLabels = [intToStrLabel[i] for i in sorted(intToStrLabel)]
        matSavePath = os.path.join(imgPath, "train-set-confusion-matrix.png")
        plotConfusionMatrix(result.metrics.confusionMatrix, classLabels,
                            matSavePath, title="Best-model CV confusion matrix")
        return result

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

    def evaluateTestSet(self, modelResult, XTest, yTest, intToStrLabels) -> (
            ClassificationMetrics):
        logger.info("Evaluating model on test set...")
        yHat = modelResult.model.predict(XTest)
        metrics = defaultClassificationMetrics(yTest, yHat)

        imgPath = self.serParams["imgPath"]
        matSavePath = os.path.join(imgPath, "test-set-confusion-matrix.png")
        classLabels = [intToStrLabels[i] for i in sorted(intToStrLabels)]
        plotConfusionMatrix(metrics.confusionMatrix, classLabels,
                            matSavePath, title="Test-set confusion matrix")

        reportModelSelection([modelResult.hyperparameters], [metrics],
                             intToStrLabels,
                             title="Test set performance")
        return metrics
