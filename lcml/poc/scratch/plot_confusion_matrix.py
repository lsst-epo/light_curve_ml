
import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from lcml.pipeline.stage.visualization import plotConfusionMatrix


def main():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel="linear", C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plotConfusionMatrix(cnf_matrix, classes=class_names, savePath=None,
                        normalize=False, title="Non-normed confusion matrix")

    # Plot normalized confusion matrix
    plotConfusionMatrix(cnf_matrix, classes=class_names, savePath=None,
                        normalize=True, title="Normalized confusion matrix")



if __name__ == "__main__":
    main()
