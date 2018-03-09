from collections import namedtuple
import time

import numpy as np
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, cross_validate

from lcml.pipeline.data_format.db_format import connFromParams, deserArray
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.data_util import attachLabels, convertClassLabels
from lcml.utils.format_util import truncatedFloat


logger = BasicLogging.getLogger(__name__)


ModelSelectionResult = namedtuple("ModelSelectionResult",
                                  ["model", "hyperparameters", "metrics"])


ClassificationMetrics = namedtuple("ClassificationMetrics",
                                   ["accuracy", "f1Overall", "f1Individual",
                                    "confusionMatrix"])


def gridSearchSelection(params):
    jobs = params["jobs"]
    # default for num estimators is 10
    estimatorsStart = params["estimatorsStart"]
    estimatorsStop = params["estimatorsStop"]

    # default for max features is sqrt(len(features))
    # for feets len(features) ~= 64 => 8
    featuresStart = params["rfFeaturesStart"]
    featuresStop = params["rfFeaturesStop"]
    return ((RandomForestClassifier(n_estimators=t, max_features=f,
                                    n_jobs=jobs),
             {"trees": t, "maxFeatures": f})
            for f in range(featuresStart, featuresStop)
            for t in range(estimatorsStart, estimatorsStop))


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
    cv = selectionParams["cv"]
    jobs = selectionParams["jobs"]
    # N.B. micro averaged preferable for imbalanced classes
    scoring = ["accuracy", "f1_micro"]

    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    query = "SELECT label from %s" % dbParams["feature_table"]
    cursor.execute(query)

    labels = [r[0] for r in cursor.fetchall()]
    labels, classToLabel = convertClassLabels(labels)

    query = "SELECT features from %s" % dbParams["feature_table"]
    features = [deserArray(r[0]) for r in cursor.execute(query)]
    # TODO potential memory issue
    # each feature array will be around 576 bytes
    # => can fit 17,361,111 feature vectors in 10GB RAM
    #
    # if this soaks up all the RAM try memory-mapped numpy array
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
    #
    # - alternatively, we can randomly select a subset:
    # choice = set(np.random.choice(setSize, subsetSize, replace=False))

    bestResult = None
    allResults = []
    maxScore = 0
    modelCount = 0
    logger.info("cross validating models...")
    for model, hyperparams in models:
        logger.info("%s", hyperparams)
        modelCount += 1
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

    conn.close()
    elapsed = time.time() - start
    logger.info("fit %s models in: %.2fs ave: %.3fs", modelCount, elapsed,
                elapsed / modelCount)
    return bestResult, allResults, classToLabel


def reportModelSelection(bestResult, allResults, classToLabel, places):
    """Reports the hyperparameters and associated metrics obtain from model
    selection."""
    reportColumns = ["Hyperparameters", "F1 (micro)", "F1 (class)", "Accuracy"]
    roundFlt = truncatedFloat(places)
    searchTable = PrettyTable(reportColumns)
    for result in allResults:
        searchTable.add_row(_resultToRow(result, classToLabel, roundFlt))

    winnerTable = PrettyTable(reportColumns)
    winnerTable.add_row(_resultToRow(bestResult, classToLabel, roundFlt))

    logger.info("Model search results...\n" + str(searchTable))
    logger.info("Winning model...\n" + str(winnerTable))


def _resultToRow(result, classToLabel, roundFlt):
    """Converts a ModelSelectionResult to a list of formatted values to be used
    as a row in a table"""
    microF1 = roundFlt % result.metrics.f1Overall
    classF1s = [(l, roundFlt % v)
                for l, v
                in attachLabels(result.metrics.f1Individual, classToLabel)]
    accuracy = roundFlt % (100 * result.metrics.accuracy)
    return [result.hyperparameters, microF1, classF1s, accuracy]
