#!/usr/bin/env python3
"""Refresher on performing cross-validation"""
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     cross_validate, train_test_split)


def getSVC():
    return svm.SVC(kernel='linear', C=1 ,random_state=0)


def main():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        test_size=0.4,
                                                        random_state=0)
    clf = getSVC().fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Basic test-train split score: %.3f" % score)

    folds = 5
    clf = getSVC()
    scores = cross_val_score(clf, iris.data, iris.target, cv=folds, n_jobs=-1)
    print("k-folds CV score: %0.3f (+/- %0.3f)" % (scores.mean(),
                                                   scores.std() * 2))


    scores = cross_val_score(clf, iris.data, iris.target, cv=folds,
                             scoring="f1_macro", n_jobs=-1)
    print("k-folds CV f1_macro score: %0.3f (+/- %0.3f)" % (scores.mean(),
                                                            scores.std() * 2))

    # N.B. common scoring types:
    # scikit-learn.org/stable/modules/model_evaluation.html
    scoring = ['precision_macro', 'recall_macro']
    clf = getSVC()
    scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                            cv=folds, return_train_score=False, n_jobs=-1)
    print("\nMultiple scores: keys: %s" % sorted(scores))
    print("- test_recall_macro: %s" % scores["test_recall_macro"])
    print("- test_precision_macro: %s" % scores["test_precision_macro"])

    # obtaining predictions by cross-validation
    predicted = cross_val_predict(clf, iris.data, iris.target, n_jobs=-1)
    acc = accuracy_score(iris.target, predicted)
    print("\nPredictions from cross-validation: %s" % acc)


if __name__ == "__main__":
    main()
