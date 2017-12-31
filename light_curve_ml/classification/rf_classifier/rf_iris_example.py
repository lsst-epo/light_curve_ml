"""Random Forest tutorial adapted from
chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html"""
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def main():
    np.random.seed(0)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # adds category column
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # assign 75% of rows to train set
    df["is_train"] = np.random.uniform(0, 1, len(df)) <= 0.75

    # train and test df's
    train, test = df[df["is_train"] == True], df[df["is_train"] == False]
    print("train obs:", len(train))
    print("test obs:", len(test))

    # names of columns that are features
    features = df.columns[:4]

    # map category names to positive ints
    y = pd.factorize(train["species"])[0]

    # train classifier
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train[features], y)

    # run clf on test
    predClasses = clf.predict(test[features])

    predProbs = clf.predict_proba(test[features])[0:10]

    preds = iris.target_names[predClasses]

    # peek results
    # print("predicted:\n%s" % preds[:5])
    # print("actual:\n%s" % test["species"].head())

    # confusion matrix
    mat = pd.crosstab(test["species"], preds, rownames=["Actual Species"],
                      colnames=["Predicted Species"])
    print(mat)

    featImp = sorted(zip(train[features], clf.feature_importances_),
                     key=lambda x: x[1], reverse=True)
    t = PrettyTable(["Feature", "Importance"])
    for f, i in featImp:
        t.add_row([f, i])
    print(t)


if __name__ == "__main__":
    main()
