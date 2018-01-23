import os
import tarfile

import numpy as np

from lcml.utils.context_util import absoluteFilePaths, joinRoot


#: Research by Kim suggests it best that light curves have at least 80 data
#: points for accurate classification
SUFFICIENT_LC_DATA = 80


def lcFilterBogus(mjds, values, errors, remove):
    """Simple light curve filter that removes bogus magnitude and error
    values."""
    return zip(*[(mjds[i], v, errors[i])
                 for i, v in enumerate(values)
                 if v not in remove and errors[i] not in remove])


def unarchiveAll(directory, ext="tar", mode="r:", remove=False):
    """Given a directory, untars all tar files found to that same dir.
    Optionally specify archive extension, compression type, and whether to
    remove archive file after unarchiving."""
    for i, f in enumerate(absoluteFilePaths(directory, ext=ext)):
        with tarfile.open(f, mode) as tar:
            tar.extractall(path=directory)

        if remove:
            os.remove(f)


def getDatasetFilePaths(datasetName, ext):
    """Returns the full paths of all dataset files in project data directory:
    ./light_curve_ml/data/
    :param datasetName - Name of specific data whose individual file paths will
    be returned
    :param ext - Required file extension of dataset files
    """
    path = joinRoot("data", datasetName)
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def convertClassLabels(classLabels):
    """Converts all class labels to integer values unique to individual
    classes. Labels are modified in-place.
    :param classLabels: Class labels for a dataset as array or list
    :return The mapping from integer to original class label for decoding
    """
    labelToInt = {v: i for i, v in enumerate(np.unique(classLabels))}
    for i in range(len(classLabels)):
        classLabels[i] = labelToInt[classLabels[i]]

    return {i: v for v, i in labelToInt.items()}


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
