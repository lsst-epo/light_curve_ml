from collections import namedtuple
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict

from lcml.utils.basic_logging import getBasicLogger


logger = getBasicLogger(__name__, __file__)


ModelSelectionResult = namedtuple("ModelSelectionResult",
                                  ["model", "hyperparameters", "metrics"])


ClassificationMetrics = namedtuple("ClassificationMetrics",
                                   ["accuracy", "f1Overall", "f1Individual",
                                    "confusionMatrix"])


def selectBestModel(models, features, labels, selectionParams):
    """Peforms k-fold cross validation on all specified models and selects model
    with highest key metric score. Returns the best model, its hyperparameters,
    and its scoring metrics.

    :param models: models with differing hyperparaters to try
    :param features: input features
    :param labels: ground truth for each feature set
    :param selectionParams: custom params
    :returns Best ModelSelectionResult and list of all ModelSelectionResults
    """
    cv = selectionParams["cv"]
    jobs = selectionParams["jobs"]

    # main output of this function
    bestResult = None
    allResults = []
    maxScore = 0

    # ancillary time costs
    fitTimes = []
    scoreTimes = []
    for model, hyperparams in models:
        fStart = time.time()
        predicted = cross_val_predict(model, features, labels, cv=cv,
                                      n_jobs=jobs)
        fitTimes.append(time.time() - fStart)

        sStart = time.time()
        accuracy = accuracy_score(labels, predicted)
        f1Overall = f1_score(labels, predicted, average="micro")
        f1Individual = f1_score(labels, predicted, average=None)
        confusionMatrix = confusion_matrix(labels, predicted)
        scoreTimes.append(time.time() - sStart)

        metrics = ClassificationMetrics(accuracy, f1Overall, f1Individual,
                                        confusionMatrix)
        result = ModelSelectionResult(model, hyperparams, metrics)
        allResults.append(result)
        if f1Overall > maxScore:
            maxScore = f1Overall
            bestResult = result

    logger.info("- average fit time: %.2fs", np.average(fitTimes))
    logger.info("- average score time: %.2fs", np.average(scoreTimes))
    return bestResult, allResults
