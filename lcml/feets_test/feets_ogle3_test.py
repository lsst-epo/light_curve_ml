from collections import Counter
import time

from feets import FeatureSpace
import numpy as np
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from lcml.common import STANDARD_INPUT_DATA_TYPES
from lcml.processing.preprocess import cleanDataset
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.context_util import absoluteFilePaths, joinRoot
from lcml.utils.data_util import unarchiveAll
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess


logger = getBasicLogger(__name__, __file__)


OGLE3_LABEL_TO_NUM = {'acep': 0, 'cep': 1, 'dpv': 2, 'dsct': 3, 'lpv': 4,
                      'rrlyr': 5}


OGLE3_NUM_TO_LABEL = {v: k for k, v in OGLE3_LABEL_TO_NUM.items()}


def _check_dim(lc):
    if lc.ndim == 1:
        lc.shape = 1, 3
    return lc


def parseOgle3Lc(filePath):
    lc = _check_dim(np.loadtxt(filePath))
    return [lc[:, 0], lc[:, 1], lc[:, 2]]


def ogle3ToLcs(dataDir, limit=float("inf")):
    """Converts all OGLE3 dat's to light curves as tuples of the form:
    (classLabel (int), time, magnitude, error). Parses class label from file
    name."""
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
        lc = parseOgle3Lc(f)
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
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)

    startTime = time.time()
    for _, tm, mag, err in lcs:
        _, ftValues = fs.extract(time=tm, magnitude=mag, error=err)
        features.append(ftValues)

    logger.info("extract in: %.02fs", time.time() - startTime)
    return features


def main():
    """Runs feets on ogle"""
    start = time.time()

    # for quick testing
    sampleLimit = 50
    trainRatio = 0.75

    dataDir = joinRoot("data/ogle3")

    unarchiveAll(dataDir, remove=True)
    lcs, categories = ogle3ToLcs(dataDir, limit=sampleLimit)
    reportClassHistogram(lcs)
    cleanLcs = cleanDataset(lcs, {float("nan")})
    reportClassHistogram(cleanLcs)

    # run train set through the feets library to get feature vectors
    allFeatures = False  # TO DO arg
    exclude = [] if allFeatures else ["CAR_mean", "CAR_sigma", "CAR_tau"]
    logger.info("Excluded features: %s", exclude)
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    cleanLcs = [(fs,) + lc for lc in cleanLcs]
    featureLabels, elapsedMin = mapMultiprocess(feetsExtract, cleanLcs)
    features = [fl[0] for fl in featureLabels]
    classLabels = [fl[1] for fl in featureLabels]

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
    print("Elapsed %.2fs" % (time.time() - start))


if __name__ == "__main__":
    main()
