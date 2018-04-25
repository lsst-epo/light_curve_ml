#!/usr/bin/env python3
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


def main():
    """From http://scikit-learn.org/stable/modules/generated/sklearn.
    model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV"""
    iris = datasets.load_iris()

    njobs = -1
    classWeight = "balanced"
    # oob_score = False  # OOB estimates are usually very pessimistic thus we
    #  recommend to use cross-validation instead and only use OOB if
    # cross-validation is too time consuming.
    estimator = RandomForestClassifier(n_jobs=njobs, class_weight=classWeight)
    param_grid = {"n_estimators": [100, 200, 300],
                  "max_features": ["sqrt", "log2", None]}
    scoring = "f1_weighted"  # "f1_micro"

    pre_dispatch = "10 * n_jobs"
    iid = True  # assume class distribution is iid

    nSplits = 5
    repeats = 1
    cv = RepeatedStratifiedKFold(n_splits=nSplits, n_repeats=repeats)

    verbose = 2
    error_score = "raise"

    clf = GridSearchCV(estimator=estimator, param_grid=param_grid,
                       scoring=scoring, pre_dispatch=pre_dispatch, iid=iid,
                       cv=cv, verbose=verbose, error_score=error_score)
    clf.fit(iris.data, iris.target)

    cvResults = clf.cv_results_
    print("result keys: %s" % list(cvResults))
    bestEstimator = clf.best_estimator_

    bestScore = clf.best_score_
    print("best score: %s" % bestScore)
    bestParams = clf.best_params_
    print("best params: %s" % bestParams)


if __name__ == "__main__":
    main()
