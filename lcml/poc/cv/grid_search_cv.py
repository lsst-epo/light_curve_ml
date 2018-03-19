from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


def main():
    """From http://scikit-learn.org/stable/modules/generated/sklearn.
    model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV"""
    iris = datasets.load_iris()

    njobs = -1
    classWeight = "balanced"
    oob_score = False # experiment
    estimator = RandomForestClassifier(n_estimators=50, max_features="sqrt",
                                       n_jobs=njobs, class_weight=classWeight,
                                       oob_score=oob_score)
    param_grid = {"n_estimators": [100],
                  "max_features": ["log2", None]}
    scoring = "f1_micro"
    # scoring = "f1_weighted"

    pre_dispatch = "10 * n_jobs"
    iid = True

    nSplits = 5
    repeats = 2
    cv = RepeatedStratifiedKFold(n_splits=nSplits, n_repeats=repeats)

    verbose = 2
    error_score = "raise"

    clf = GridSearchCV(estimator=estimator, param_grid=param_grid,
                       scoring=scoring, pre_dispatch=pre_dispatch, iid=iid,
                       cv=cv, verbose=verbose, error_score=error_score)
    clf.fit(iris.data, iris.target)

    cvResults = clf.cv_results_
    print(cvResults)
    bestEstimator = clf.best_estimator_
    bestScore = clf.best_score_
    bestParams = clf.best_params_


if __name__ == "__main__":
    main()
