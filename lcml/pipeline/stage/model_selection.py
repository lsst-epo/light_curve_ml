from collections import namedtuple
from datetime import timedelta
import time

import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, GridSearchCV, RepeatedStratifiedKFold

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.dataset_util import attachLabels
from lcml.utils.format_util import truncatedFloat


logger = BasicLogging.getLogger(__name__)


_SCORING_TYPES = ["accuracy", "f1_micro", "f1_macro", "f1_weighted"]


ModelSelectionResult = namedtuple("ModelSelectionResult",
                                  ["model", "hyperparameters", "metrics"])


class ClassificationMetrics:
    """Selected metrics for evaluating light curve classification"""
    def __init__(self, accuracy, f1Micro, f1Macro, f1Weighted, confusionMatrix):
        self.accuracy = accuracy
        self.f1Micro = f1Micro
        self.f1Macro = f1Macro
        self.f1Weighted = f1Weighted
        self.confusionMatrix = confusionMatrix


def getClassificationMetrics(y, yHat):
    # TODO research average=None vs. average="macro"
    return ClassificationMetrics(accuracy_score(y, yHat),
                                 f1_score(y, yHat, average="micro"),
                                 f1_score(y, yHat, average="macro"),
                                 f1_score(y, yHat, average="weighted"),
                                 confusion_matrix(y, yHat))


def randomForestGridSearch(params):
    """Returns a generator of `RandomForestClassifier` params and hyperparams
    as a kwarg dict"""
    jobs = params["jobs"]
    classWeight = params["classWeight"]

    # default for num estimators is 10
    if "n_trees" in params:
        treesAxis = params["n_trees"]
    else:
        treesStart = params["treesStart"]
        treesStop = params["treesStop"]
        treesAxis = range(treesStart, treesStop)

    if "max_features" in params:
        featuresAxis = params["max_features"]
    else:
        # default for max features is sqrt(len(features))
        # for feets len(features) ~= 64 => 8
        featuresStart = params["maxFeaturesStart"]
        featuresStop = params["maxFeaturesStop"]
        featuresAxis = range(featuresStart, featuresStop)

    return (({"n_estimators": t, "max_features": f, "n_jobs": jobs,
              "class_weight": classWeight})
            for f in featuresAxis for t in treesAxis)


def gridSearchCv(modelClass, X, y, folds, repeats):
    # TODO meld this with supervised pipeline, pass in params, compute metrics
    # as in `selectBestModel` which may require splitting in selection and
    #  training
    njobs = -1
    classWeight = "balanced"
    estimator = modelClass(n_jobs=njobs, class_weight=classWeight)

    param_grid = {"n_estimators": [100, 200, 300],
                  "max_features": ["sqrt", "log2", None]}
    scoring = "f1_weighted"
    pre_dispatch = "10 * n_jobs"
    iid = True  # assume class distribution is iid
    verbose = 2
    error_score = "raise"

    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid,
                       scoring=scoring, pre_dispatch=pre_dispatch, iid=iid,
                       cv=cv, verbose=verbose, error_score=error_score)
    clf.fit(X, y)

    cvResults = clf.cv_results_
    print("result keys: %s" % list(cvResults))

    bestScore = clf.best_score_
    print("best score: %s" % bestScore)
    bestParams = clf.best_params_
    print("best params: %s" % bestParams)

    metrics = None
    return ModelSelectionResult(clf.best_estimator_, clf.best_params_, metrics)


def selectBestModel(modelClass, hyperparamsItr, X, y, folds, repeats,
                    selectionMetricName="test_f1_weighted"):
    """Peforms k-fold cross validation on all specified models and selects model
    with highest selection metric. Returns the best `ModelSelectionResult`
    trained and evaluated on the train set as well as all `ModelSelectionResult`
    instance obtained from cross validation.

    :param modelClass: class of model on which selection is performed
    :param hyperparamsItr: hyperparameters to be tried
    :param X: training set of feature vectors
    :param y: training set of class labels
    :param folds: number of folds used in cv
    :param repeats: number of CV repetitions with different randomization
    :param selectionMetricName: the name of the metric on which the selection is
    based
    :returns Best ModelSelectionResult and list of all ModelSelectionResults
    """
    start = time.time()
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
    bestHyperparams = None
    allResults = []
    maxScore = 0
    modelCount = None
    logger.info("Selecting %s model using %s-fold cross-validation "
                "repeat=%s on train set...", modelClass.__name__, folds,
                repeats)
    for modelCount, kwargs in enumerate(hyperparamsItr):
        cvStart = time.time()
        model = modelClass(**kwargs)
        scores = cross_validate(model, X, y, scoring=_SCORING_TYPES, cv=cv,
                                n_jobs=-1)

        # average metrics across folds
        accuracy = np.average(scores["test_accuracy"])
        f1Micro = np.average(scores["test_f1_micro"])
        f1Macro = np.average(scores["test_f1_macro"])
        f1Weight = np.average(scores["test_f1_weighted"])
        mets = ClassificationMetrics(accuracy, f1Micro, f1Macro, f1Weight, None)
        allResults.append(ModelSelectionResult(model, kwargs, mets))

        # update max
        selectMetric = np.average(scores[selectionMetricName])
        if selectMetric > maxScore:
            maxScore = selectMetric
            bestHyperparams = kwargs

        logger.info("%s in %.2fs", kwargs, time.time() - cvStart)

    if not modelCount:
        raise ValueError("No hyperparameters specified")

    elapsed = timedelta(seconds=time.time() - start)
    logger.info("fit %s models in: %s ave: %s", modelCount, elapsed,
                elapsed / modelCount)

    # train winner on entire training set
    logger.info("training winning model on train set...")
    bestModel = modelClass(**bestHyperparams)
    bestModel.fit(X, y)
    yHat = bestModel.predict(X)
    bestMetrics = getClassificationMetrics(y, yHat)
    bestResult = ModelSelectionResult(bestModel, bestHyperparams, bestMetrics)
    return bestResult, allResults


_REPORT_COLS = ["Hyperparameters", "F1 micro", "F1 macro", "F1 weighted",
                "Accuracy"]


def reportModelSelection(hyperparamsList, metricsList, classToLabel,
                         places=3, title=None):
    """Reports the hyperparameters and associated metrics obtain from model
    selection."""
    t = PrettyTable(_REPORT_COLS)
    if title:
        t.title = title
    roundedFloat = truncatedFloat(places)
    for i, hyperparams in enumerate(hyperparamsList):
        mets = metricsList[i]
        t.add_row(_resultToRow(hyperparams, mets, classToLabel, roundedFloat))

    logger.info("\n%s", t)


def _resultToRow(hyperparameters, metrics, classToLabel, roundFlt):
    """Converts a ModelSelectionResult to a list of formatted values to be used
    as a row in a table"""
    f1Micro = roundFlt % metrics.f1Micro
    f1Macro = roundFlt % metrics.f1Macro
    if isinstance(f1Macro, (np.ndarray, list)):
        f1Macro = [(l, roundFlt % v)
                     for l, v
                     in attachLabels(metrics.f1Macro, classToLabel)]
    f1Weighted = roundFlt % metrics.f1Weighted
    accuracy = roundFlt % (100 * metrics.accuracy)
    return [hyperparameters, f1Micro, f1Macro, f1Weighted, accuracy]
