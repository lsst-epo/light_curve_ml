from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.stage.model_selection import (reportModelSelection,
                                                 selectBestModel)
from lcml.pipeline.stage.persistence import loadModels, saveModel
from lcml.pipeline.stage.visualization import contourPlot, plotConfusionMatrix
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


class SupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self):
        """Runs the supervised portion of a batch machine learning pipeline.
        Performs following stages:
        4) obtain models and peform model selection
        5) serialize winning model to disk
        6) report metrics
        """
        models = None
        modelLoadPath = self.serParams["modelLoadPath"]
        if modelLoadPath:
            models = loadModels(modelLoadPath)
        if not models:
            models = self.selectionFcn(self.selectionParams)
        bestResult, allResults, classToLabel = (
            selectBestModel(models, self.selectionParams, self.dbParams)
        )
        if self.serParams["modelSavePath"]:
            saveModel(bestResult, self.serParams["modelSavePath"],
                      self.conf, classToLabel)

        reportModelSelection(bestResult, allResults, classToLabel,
                             self.globalParams.get("places", 3))

        # plot effects of hyperparameters on F1-micro
        x, y, z = zip(*[(r.hyperparameters["trees"],
                         r.hyperparameters["maxFeatures"],
                         r.metrics.f1Overall)
                        for r in allResults])
        savePath = "/Users/ryanjmccall/code/light_curve_ml/models/macho/hyper.png"
        contourPlot(x, y, z, savePath)

        logger.info("Integer class label mapping %s", classToLabel)
        classLabels = [classToLabel[i] for i in sorted(classToLabel)]
        plotConfusionMatrix(bestResult.metrics.confusionMatrix, classLabels,
                            self.serParams["confusionMatrixSavePath"])
