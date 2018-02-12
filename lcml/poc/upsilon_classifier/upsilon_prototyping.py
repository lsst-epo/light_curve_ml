#!/usr/bin/env python
import argparse
import os
import logging
import time
import traceback

import numpy as np
from prettytable import PrettyTable
import upsilon

from lcml.pipeline.preprocess import lcFilterBogus, SUFFICIENT_LC_DATA

_GARBAGE_VALUES = {float("nan"), float("-inf"), float("inf")}
_MACHO_REMOVE = _GARBAGE_VALUES.union({-99.0})


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _toFloat(v):
    f = float(v)
    if f in _GARBAGE_VALUES:
        raise ValueError("Garbage value")

    return f


def confidenceInterval(data, numStds):
    mean = np.mean(data)
    std = np.std(data)
    return mean - numStds * std, mean + numStds * std


def removeOutliers(values, errors, numStds=3):
    """Given a time series of values and their associated errors returns a copy
    of both arrays with the outliers removed. If either the value or the error
    extends beyond the specified number of standard deviations, the entire
    point (value and error) is dropped."""
    valLwr, valUpr = confidenceInterval(values, numStds)
    errLwr, errUpr = confidenceInterval(errors, numStds)
    return zip(*[(v, errors[i]) for i, v in enumerate(values)
                 if valLwr < v < valUpr and errLwr < errors[i] < errUpr])


def classifyLightCurve(mjds, magnitudes, errors, rfModel, nThreads=4):
    """Given a light curve time series of Modified Julian Date, magnitude, and
    magnitude error, extract the key features and classify returning a class
    label, probability estimate of poc, and flag. If flag is 1, the
    poc is "suspicious." (either 1) period is in period alias, or
    2) period SNR is lower than 20)

    :param mjds: ndarray
    :param magnitudes: ndarray
    :param errors: ndarray
    :param rfModel: RandomForest model

    :return class label, probability, and flag
    """
    start = time.time()
    e_features = upsilon.ExtractFeatures(mjds, magnitudes, errors,
                                         n_threads=nThreads)
    e_features.run()
    features = e_features.get_features()

    # for 500 data points, this fcn takes 0.22 + 0.11 = 0.33s
    logger.info("Computed features in: %.2fs", time.time() - start)

    # Classify the light curve; takes around 0.1s
    return upsilon.predict(rfModel, features)


def loadMachoData(trainData, report=True):
    mjds = []
    redMags = []
    redErrors = []
    blueMags = []
    blueErrors = []
    for line in trainData:
        try:
            mjd = _toFloat(line[4])
            rMag = _toFloat(line[9])
            rErr = _toFloat(line[10])
            bMag = _toFloat(line[24])
            bErr = _toFloat(line[25])
        except ValueError:
            print("bad value: %s" % line)
            continue

        mjds.append(mjd)
        redMags.append(rMag)
        redErrors.append(rErr)
        blueMags.append(bMag)
        blueErrors.append(bErr)

    redMjds, clRedMags, clRedErrors = lcFilterBogus(mjds, redMags,
                                                    redErrors, _MACHO_REMOVE)
    blueMjds, clBlueMags, clBlueErrors = lcFilterBogus(mjds, blueMags,
                                                       blueErrors,
                                                       _MACHO_REMOVE)
    if report:
        t = PrettyTable(field_names=["type", "original", "removed",
                                     "final"])
        t.add_row(["red", len(redMags), len(redMags) - len(clRedMags),
                   len(clRedMags)])
        t.add_row(["blue", len(blueMags), len(blueMags) - len(clBlueMags),
                   len(clBlueMags)])
        logger.info("\n%s", t)

    return [[np.array(redMjds), np.array(clRedMags), np.array(clRedErrors)],
            [np.array(blueMjds), np.array(clBlueMags), np.array(clBlueErrors)]]


def runDataset(dataDir, randomForestModel, outDir, nThreads=4, maxRows=100):
    outlines = [",".join(["file", "label", "prob", "flag\n"])]
    for fieldDir in sorted(os.listdir(dataDir)):
        fieldPath = os.path.join(dataDir, fieldDir)
        if not os.path.isdir(fieldPath):
            continue

        for file in sorted(os.listdir(fieldPath)):
            filePath = os.path.join(fieldPath, file)
            logger.info("\nLoading datafile %s", filePath)
            trainData = np.genfromtxt(filePath, delimiter=';', dtype=None,
                                      max_rows=maxRows)
            for mjds, mags, errors in loadMachoData(trainData):
                if len(mjds) < SUFFICIENT_LC_DATA:
                    logger.info("Skipping due to insufficient data %s < %s",
                                len(mjds), SUFFICIENT_LC_DATA)
                    continue

                try:
                    label, prob, flag = classifyLightCurve(mjds, mags, errors,
                                                           randomForestModel,
                                                           nThreads=nThreads)
                except ValueError:
                    traceback.print_exc()
                    continue

                logger.info("Classification: %s probability: %s flag: %s",
                            label, prob, flag)

                outlines.append(",".join([file, label, str(prob),
                                          str(flag) + "\n"]))

    with open(os.path.join(outDir, "batch-results.csv"), "w") as outFile:
        outFile.writelines(outlines)


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", type=int,
                        help="number of threads dedicated to feature (period) "
                             "data")
    parser.add_argument("-r", "--rows", type=int,
                        help="limit on initial portion of timeseries processed")
    return parser.parse_args()


def main():
    start = time.time()
    args = _getArgs()
    dataset = "macho"
    dataDir = os.path.join(os.environ.get("LSST"), "data", dataset)
    outDir = os.path.join(os.environ.get("LSST"), "results", dataset)
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    logger.info("Loading RF classifier...")
    randomForestModel = upsilon.load_rf_model()
    runDataset(dataDir, randomForestModel, outDir, args.threads, args.rows)
    logger.info("finished in: %.2fs", time.time() - start)


if __name__ == "__main__":
    main()
