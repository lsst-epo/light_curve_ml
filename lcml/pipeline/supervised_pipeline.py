from datetime import timedelta
import os
import time

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.ml_pipeline_conf import MlPipelineConf
from lcml.pipeline.stage.model_selection import (ClassificationMetrics,
                                                 defaultClassificationMetrics,
                                                 ModelSelectionResult,
                                                 reportModelSelection)
from lcml.pipeline.stage.persistence import loadModelAndHyperparms
from lcml.pipeline.stage.visualization import plotConfusionMatrix
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


class SupervisedPipeline(BatchPipeline):
    def __init__(self, conf: MlPipelineConf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self, XTrain, yTrain,
                            intToStrLabel) -> ModelSelectionResult:
        """Runs the supervised portion of a batch machine learning pipeline.
        Loads a model and its hyperparams from disk if specified or performs
        model selection and to obtain a ModelSelectionResult.
        """
        modelLoadPath = self.serStage.params["modelLoadPath"]
        if modelLoadPath:
            model, hyperparams = loadModelAndHyperparms(modelLoadPath)
            result = ModelSelectionResult(model, hyperparams, None)
        else:
            start = time.time()
            params = self.searchStage.params
            result = self.searchStage.fcn(params["model"], XTrain, yTrain,
                                          params["cv"], params["gridSearch"],
                                          self.globalParams)
            logger.info("search completed in: %s",
                        timedelta(seconds=time.time() - start))

            roundPlaces = self.globalParams["places"]
            reportModelSelection([result.hyperparameters], [result.metrics],
                                 intToStrLabel, roundPlaces,
                                 title="Best result")

        imgPath = self.serStage.params["imgPath"]
        classLabels = [intToStrLabel[i] for i in sorted(intToStrLabel)]
        matSavePath = os.path.join(imgPath, "train-set-confusion-matrix.png")
        plotConfusionMatrix(result.metrics.confusionMatrix, classLabels,
                            matSavePath, title="Best-model CV confusion matrix")
        return result

    def evaluateTestSet(self, modelResult, XTest, yTest, intToStrLabels) -> (
            ClassificationMetrics):
        logger.info("Evaluating model on test set...")
        yHat = modelResult.model.predict(XTest)
        metrics = defaultClassificationMetrics(yTest, yHat)

        imgPath = self.serStage.params["imgPath"]
        matSavePath = os.path.join(imgPath, "test-set-confusion-matrix.png")
        classLabels = [intToStrLabels[i] for i in sorted(intToStrLabels)]
        plotConfusionMatrix(metrics.confusionMatrix, classLabels,
                            matSavePath, title="Test-set confusion matrix")

        reportModelSelection([modelResult.hyperparameters], [metrics],
                             intToStrLabels,
                             title="Test set performance")
        return metrics
