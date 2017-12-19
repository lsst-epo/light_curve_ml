#: Research by Kim has uncovered that light curves should have at least 80
#: data points to be classifiable
import os

import numpy as np

from light_curve_ml.utils import context_util


SUFFICIENT_LC_DATA = 80


def removeMachoOutliers(mjds, values, errors, remove=-99.0):
    """Simple bogus value filter for MACHO magnitudes and errors."""
    return zip(*[(mjds[i], v, errors[i])
                 for i, v in enumerate(values)
                 if v != remove and errors[i] != remove])


def getDatasetFilePaths(datasetName, ext):
    """Returns the full paths of all dataset files in project data directory:
    ./light_curve_ml/data/
    :param datasetName - Name of specific data whose individual file paths will
    be returned
    :param ext - Required file extension of dataset files
    """
    path = context_util.joinRoot("data", datasetName)
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def reportDataset(dataset, labels=None):
    """Reports the characteristics of a dataset"""
    size = len(dataset)
    dataSizes = [len(x) for x in dataset]
    minSize = min(dataSizes)
    maxSize = max(dataSizes)
    ave = np.average(dataSizes)
    std = float(np.std(dataSizes))
    print("_Dataset Report_")
    print("size: %s \nmin: %s \nave: %.02f (%.02f) \nmax: %s" % (
        size, minSize, ave, std, maxSize))
    if labels:
        print("Unique labels: %s" % sorted(np.unique(labels)))