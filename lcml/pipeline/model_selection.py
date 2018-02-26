from collections import namedtuple
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, cross_validate

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


ModelSelectionResult = namedtuple("ModelSelectionResult",
                                  ["model", "hyperparameters", "metrics"])


ClassificationMetrics = namedtuple("ClassificationMetrics",
                                   ["accuracy", "f1Overall", "f1Individual",
                                    "confusionMatrix"])


def selectBestModel(models, features, labels, selectionParams):
    """Peforms k-fold cross validation on all specified models and selects model
    with highest f1_micro score. Returns the best model, its hyperparameters,
    and its scoring metrics including accuracy, f1_micro, individual class f1,
    and confusion matrix

    :param models: models with differing hyperparaters to try
    :param features: input features
    :param labels: ground truth for each feature set
    :param selectionParams: custom params
    :returns Best ModelSelectionResult and list of all ModelSelectionResults
    """
    start = time.time()
    cv = selectionParams["cv"]
    jobs = selectionParams["jobs"]
    # N.B. micro averaged preferable for imbalanced classes
    scoring = ["accuracy", "f1_micro"]

    # main output of this function
    bestResult = None
    allResults = []
    maxScore = 0
    for model, hyperparams in models:
        scores = cross_validate(model, features, labels, scoring=scoring, cv=cv,
                                n_jobs=jobs)
        accuracy = np.average(scores["test_accuracy"])
        f1Overall = np.average(scores["test_f1_micro"])

        # cannot compute these two from 'cross_validate' results
        predicted = cross_val_predict(model, features, labels, cv=cv,
                                      n_jobs=jobs)
        f1Individual = f1_score(labels, predicted, average=None)
        confusionMatrix = confusion_matrix(labels, predicted)

        metrics = ClassificationMetrics(accuracy, f1Overall, f1Individual,
                                        confusionMatrix)
        result = ModelSelectionResult(model, hyperparams, metrics)
        allResults.append(result)
        if f1Overall > maxScore:
            maxScore = f1Overall
            bestResult = result

    elapsed = time.time() - start
    logger.info("fit %s models in: %.2fs ave: %.3fs", len(models),
                elapsed, elapsed / len(models))
    return bestResult, allResults
