from collections import Counter
import os
import tarfile
import time

import numpy as np
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from feets import FeatureSpace
from feets.datasets.ogle3 import load_OGLE3_catalog
from lcml.common import STANDARD_DATA_TYPES
from lcml.processing.preprocess import preprocessLc
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.context_util import absoluteFilePaths, joinRoot


logger = getBasicLogger(__name__, __file__)


OGLE3_LABEL_TO_NUM = {'acep': 0, 'cep': 1, 'dpv': 2, 'dsct': 3, 'lpv': 4,
                      'rrlyr': 5}


OGLE3_NUM_TO_LABEL = {v: k for k, v in OGLE3_LABEL_TO_NUM.items()}


def untarDir(targetDir, mode="r:", remove=False):
    """Given a directory, untars all tar files found to that same dir.
    Optionally specify compression type and whether to remove .tar file."""
    for i, f in enumerate(absoluteFilePaths(targetDir, ext="tar")):
        with tarfile.open(f, mode) as tar:
            tar.extractall(path=targetDir)

        if remove:
            os.remove(f)


def _check_dim(lc):
    if lc.ndim == 1:
        lc.shape = 1, 3
    return lc


def parseOgle3Lc(filePath, category, metadata=False):
    """Converts an OGLE3 data file to feets.datasets.base.Bunch"""
    # ogle3_id = filePath.split("/")[-1].split(".")[0]
    # if metadata:
    #     cat = load_OGLE3_catalog()
    #     metadata = cat[cat.ID == ogle3_id].iloc[0].to_dict()
    #     del cat
    #
    lc = _check_dim(np.loadtxt(filePath))
    return [lc[:, 0], lc[:, 1], lc[:, 2]]


def ogle3ToLcs(dataDir, limit=float("inf")):
    """Converts all OGLE3 dat's to feets.datasets.base.Bunch objects"""
    uniqueCats = set()
    lcs = []
    for i, f in enumerate(absoluteFilePaths(dataDir, ext="dat")):
        if i == limit:
            break

        fileName = f.split("/")[-1]
        fnSplits = fileName.split("-")
        if len(fnSplits) > 2:
            category = fnSplits[2].lower()
        else:
            logger.warning("file name lacks category! %s", fileName)
            continue

        uniqueCats.add(category)
        catCode = OGLE3_LABEL_TO_NUM[category]
        lc = parseOgle3Lc(f, category)
        if lc:
            lcs.append([catCode] + lc)

    return lcs, uniqueCats


def reportClassHistogram(lcs):
    c = Counter([OGLE3_NUM_TO_LABEL[lc[0]] for lc in lcs])
    total = float(len(lcs))
    t = PrettyTable(["category", "count", "percentage"])
    t.align = "l"
    for k, v in sorted(c.items(), key=lambda x: x[1], reverse=True):
        t.add_row([k, v, "{:.2%}".format(v / total)])

    logger.info("\n" + str(t))


def extractFeatures(lcs):
    features = list()
    exclude = [] if False else ["CAR_mean", "CAR_sigma", "CAR_tau"]
    logger.info("Excluded features: %s", exclude)
    fs = FeatureSpace(data=STANDARD_DATA_TYPES, exclude=exclude)

    startTime = time.time()
    for _, tm, mag, err in lcs:
        _, ftValues = fs.extract(time=tm, magnitude=mag, error=err)
        features.append(ftValues)

    logger.info("extract in: %.02fs", time.time() - startTime)
    return features


def main():
    """ abc """
    # download all ogle3 data
    # copy tars to lcml data dir
    # parse out the label from file names, lowercase it, convert to number
    # associate label to the lc
    # compute class label histogram
    start = time.time()
    limit = 50
    trainRatio = 0.75

    dataDir = joinRoot("data/ogle3")
    untarDir(dataDir, remove=True)
    lcs, categories = ogle3ToLcs(dataDir, limit=limit)

    # TODO FIXME
    # preprocessLc(timeData, magData, errorData, remove=(-99.0,), stdLimit=5,
    #              errorLimit=3)
    # TODO
    # write a small fcn to preprocess a list of lc's and report thrown away

    reportClassHistogram(lcs)

    # run train set through the feets library to get feature vectors
    classLabels = [lc[0] for lc in lcs]
    features = extractFeatures(lcs)

    # create a test and train set
    xTrain, xTest, yTrain, yTest = train_test_split(features, classLabels,
                                                    train_size=trainRatio)

    # Train and Test dataset size details
    print("\nTrain & Test sizes")
    print("Train_x Shape: ", len(xTrain))
    print("Train_y Shape: ", len(yTrain))
    print("Test_x Shape: ", len(xTest))
    print("Test_y Shape: ", len(yTest))

    # train RF on train set feature vectors
    model = RandomForestClassifier()
    model.fit(xTrain, yTrain)
    # TODO pickle the RF to disk
    # http://scikit-learn.org/stable/modules/model_persistence.html

    trainPredictions = model.predict(xTrain)
    testPredictions = model.predict(xTest)

    # accuracy
    print("Train accuracy: ", accuracy_score(yTrain, trainPredictions))
    print("Test accuracy: ", accuracy_score(yTest, testPredictions))
    print("Confusion: ", confusion_matrix(yTest, testPredictions))

    # TODO
    # research performance metrics from Kim's papers
    # record performance and time to process
    # create CV set and try RF variations
    print("Elapsed %s" % (time.time() - start))


if __name__ == "__main__":
    main()
