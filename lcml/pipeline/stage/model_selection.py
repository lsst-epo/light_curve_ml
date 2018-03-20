from collections import namedtuple
import time

import numpy as np
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (cross_val_predict, cross_validate,
                                     RepeatedStratifiedKFold)

from lcml.pipeline.database.sqlite_db import connFromParams, selectFeatures
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.dataset_util import attachLabels, convertClassLabels
from lcml.utils.format_util import truncatedFloat


logger = BasicLogging.getLogger(__name__)


ModelSelectionResult = namedtuple("ModelSelectionResult",
                                  ["model", "hyperparameters", "metrics"])


ClassificationMetrics = namedtuple("ClassificationMetrics",
                                   ["accuracy", "f1Micro", "f1Macro",
                                    "f1Weighted", "confusionMatrix"])


def gridSearchSelection(params):
    """Returns a generator of random forest classifiers for a grid of the two
    critical hyperparameters."""
    jobs = params["jobs"]

    # default for num estimators is 10
    if "treesValues" in params:
        treesAxis = params["treesValues"]
    else:
        treesStart = params["treesStart"]
        treesStop = params["treesStop"]
        treesAxis = range(treesStart, treesStop)

    if "maxFeaturesValues" in params:
        featuresAxis = params["maxFeaturesValues"]
    else:
        # default for max features is sqrt(len(features))
        # for feets len(features) ~= 64 => 8
        featuresStart = params["maxFeaturesStart"]
        featuresStop = params["maxFeaturesStop"]
        featuresAxis = range(featuresStart, featuresStop)

    classWeight = params["classWeight"]
    return ((RandomForestClassifier(n_estimators=t, max_features=f,
                                    n_jobs=jobs, class_weight=classWeight),
             {"trees": t, "maxFeatures": f})
            for f in featuresAxis for t in treesAxis)


_SCORING_TYPES = ["accuracy", "f1_micro", "f1_weighted"]


def selectBestModel(models, selectionParams, dbParams):
    """Peforms k-fold cross validation on all specified models and selects model
    with highest f1_micro score. Returns the best model, its hyperparameters,
    and its scoring metrics including accuracy, f1_micro, individual class f1,
    and confusion matrix

    :param models: models having a range of hyperparaters to be tried
    :param selectionParams: params governing model selection
    :param dbParams: params specifying database
    :returns Best ModelSelectionResult and list of all ModelSelectionResults
    """
    start = time.time()
    nSplits = selectionParams["folds"]
    repeats = selectionParams["repeats"]
    cv = RepeatedStratifiedKFold(n_splits=nSplits, n_repeats=repeats)

    jobs = selectionParams["jobs"]

    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    query = "SELECT label from %s" % dbParams["feature_table"]
    cursor.execute(query)

    labels = [r[0] for r in cursor.fetchall()]
    labels, classToLabel = convertClassLabels(labels)
    features = selectFeatures(cursor, dbParams)

    bestResult = None
    allResults = []
    maxScore = 0
    modelCount = 0
    logger.info("cross validating models...")
    for model, hyperparams in models:
        cvStart = time.time()
        modelCount += 1
        scores = cross_validate(model, features, labels, scoring=_SCORING_TYPES,
                                cv=cv, n_jobs=jobs)

        # average across folds
        accuracy = np.average(scores["test_accuracy"])
        f1Micro = np.average(scores["test_f1_micro"])
        f1Weighted = np.average(scores["test_f1_weighted"])

        # cannot compute these two from 'cross_validate' results
        predicted = cross_val_predict(model, features, labels, cv=cv,
                                      n_jobs=jobs)
        f1Macro = f1_score(labels, predicted, average=None)
        confusionMatrix = confusion_matrix(labels, predicted)

        metrics = ClassificationMetrics(accuracy, f1Micro, f1Macro, f1Weighted,
                                        confusionMatrix)
        result = ModelSelectionResult(model, hyperparams, metrics)
        allResults.append(result)
        if f1Weighted > maxScore:
            maxScore = f1Weighted
            bestResult = result

        logger.info("%s in %.2fs", hyperparams, time.time() - cvStart)

    conn.close()
    elapsed = time.time() - start
    if not modelCount:
        raise ValueError("No models specified")

    logger.info("fit %s models in: %.2fs ave: %.3fs", modelCount, elapsed,
                elapsed / modelCount)
    return bestResult, allResults, classToLabel


_REPORT_COLS = ["Hyperparameters", "F1 micro", "F1 macro", "F1 weighted",
                "Accuracy"]


def reportModelSelection(bestResult, allResults, classToLabel, places):
    """Reports the hyperparameters and associated metrics obtain from model
    selection."""
    roundFlt = truncatedFloat(places)
    searchTable = PrettyTable(_REPORT_COLS)
    for result in allResults:
        searchTable.add_row(_resultToRow(result, classToLabel, roundFlt))

    winnerTable = PrettyTable(_REPORT_COLS)
    winnerTable.add_row(_resultToRow(bestResult, classToLabel, roundFlt))

    logger.info("Model search results...\n" + str(searchTable))
    logger.info("Winning model...\n" + str(winnerTable))


def _resultToRow(result, classToLabel, roundFlt):
    """Converts a ModelSelectionResult to a list of formatted values to be used
    as a row in a table"""
    f1Micro = roundFlt % result.metrics.f1Overall
    f1Individ = [(l, roundFlt % v)
                 for l, v
                 in attachLabels(result.metrics.f1Individual, classToLabel)]
    f1Weighted = roundFlt % result.metrics.f1Weighted
    accuracy = roundFlt % (100 * result.metrics.accuracy)
    return [result.hyperparameters, f1Micro, f1Individ, f1Weighted, accuracy]
