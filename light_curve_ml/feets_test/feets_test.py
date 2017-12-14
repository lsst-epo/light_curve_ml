"""Testing out Feets on MACHO."""
import os
import time

from feets import datasets, FeatureSpace, preprocess
from feets.datasets.base import Bunch
from feets.extractors.core import DATA_TIME, DATA_MAGNITUDE, DATA_ERROR
from matplotlib import pyplot as plt
import numpy as np
from prettytable import PrettyTable

from light_curve_ml.utils import context
from light_curve_ml.utils.basic_logging import getBasicLogger
from light_curve_ml.utils.data import (removeMachoOutliers,
                                       SUFFICIENT_LC_DATA)


logger = getBasicLogger(__name__, __file__)


#: all data types accepted by the library
ALL_DATA_TYPES = ["time", "magnitude", "error", "magnitude2", "aligned_time",
                  "aligned_magnitude", "aligned_error", "aligned_magnitude2",
                  "aligned_error2"]


#: LC bands contain these time series datums
LC_DATA = (DATA_TIME, DATA_MAGNITUDE, DATA_ERROR)


def parseCleanSeries(filePaths):
    """Parses the given MACHO LC file into a list of noise-removed light curves
    in red and blue bands."""
    lcs = []
    rSuccess = 0
    bSuccess = 0
    for fp in filePaths:
        # Data format - csv with header
        # 0-class, 1-fieldid, 2-tileid, 3-seqn, 4-obsid, 5-dateobs, 6-rmag,
        # 7-rerr, 8-bmag, 9-berr
        data = np.loadtxt(fp, delimiter=",", skiprows=1)
        field = fp.split("/")[-1].split(".")[0].split("_")[-1]

        # Since each file contains multiple tileids the seqn values may not be
        # unique until data is first split into individual tileids
        tiles = np.unique(data[:, 2])
        for tile in sorted(tiles):
            tileData = data[np.where(data[:, 2] == tile)]
            sequences = np.unique(tileData[:, 3])
            for seq in sorted(sequences):
                logger.info("field: %s tile: %s seqn: %s", field, tile, seq)
                series = tileData[np.where(tileData[:, 3] == seq)]
                cat = series[0, 0]

                # sort by mjd time to be sure
                series = series[series[:, 5].argsort()]
                rBunch = parseMachoBunch(series[:, 5], series[:, 6],
                                         series[:, 7])
                if rBunch:
                    rSuccess += 1

                bBunch = parseMachoBunch(series[:, 5], series[:, 8],
                                         series[:, 9])
                if bBunch:
                    bSuccess += 1

                lcs.append(Bunch(field=field, tile=tile, sequence=seq,
                                 category=cat, data=LC_DATA,
                                 bands=Bunch(r=rBunch, b=bBunch)))

    logger.info("r success: %.02fs b success: %.02fs",
                100.0 * rSuccess / len(filePaths),
                100.0 * bSuccess / len(filePaths))
    return lcs


def parseMachoBunch(timeData, magData, errorData):
    # removes -99's endemic to MACHO
    tm, mag, err = removeMachoOutliers(timeData, magData, errorData,
                                       remove=-99.0)
    remove1 = len(timeData) - len(tm)
    if remove1:
        logger.info("macho remove %s", remove1)

    if len(tm) < SUFFICIENT_LC_DATA:
        logger.warning("insufficient: %s after removing -99", len(tm))
        return None

    # removes statistical outliers
    _tm, _mag, _err = preprocess.remove_noise(tm, mag, err)
    remove2 = len(tm) - len(_tm)
    if remove2:
        logger.info("stats remove %s", remove2)

    if len(_tm) < SUFFICIENT_LC_DATA:
        logger.warning("insufficient: %s after statistical outliers removed",
                       len(_tm))
        # return None  # TODO test with just -99 removed first

    return Bunch(**{DATA_TIME: _tm, DATA_MAGNITUDE: _mag, DATA_ERROR: _err})


def absoluteFilePaths(directory):
    return [os.path.abspath(os.path.join(dirPath, f))
            for dirPath, _, fileNames in os.walk(directory)
                for f in fileNames
                     if f != ".DS_Store"]


def machoTest():
    dataDir = context.joinRoot("data/macho/raw")
    absPaths = absoluteFilePaths(dataDir)
    lcs = parseCleanSeries(absPaths)
    # TODO simple fcn to compute min, max, ave and std for red and blue bands
    print(len(lcs))

    # fileName = "c1_f1.csv"
    # bunches = parseCleanSeries(fileName)
    # for lc in bunches:
    #     plotLc2(lc)


def plotLc2(lc):
    f = plt.figure(1)
    plt.plot(lc.bands.B.time, lc.bands.B.magnitude, "*-", alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.gca().invert_yaxis()
    f.show()


def plotLc(lc):
    f = plt.figure(1)
    plt.plot(lc.bands.B.time, lc.bands.B.magnitude, "*-", alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.gca().invert_yaxis()
    f.show()


def tutorial():
    plot = 0
    lc = datasets.load_MACHO_example()
    if plot:
        plotLc(lc)

    # 69 features with ALL_DATA_TYPES in 6.15s
    # 64 features with time, magnitude, error in 6.14s
    # 58 features with time, magnitude in 3.3s
    # 22 features with magnitude in 0.02s
    basicData = ["time", "magnitude", "error"]
    basicData = ["time", "magnitude"]
    basicData = ["magnitude"]

    start = time.time()

    # remove points beyond 5 stds of the mean
    tm, mag, error = preprocess.remove_noise(**lc.bands.B)
    tm2, mag2, error2 = preprocess.remove_noise(**lc.bands.R)

    aTime, aMag, aMag2, aError, aError2 = preprocess.align(tm, tm2, mag,
                                                           mag2, error, error2)
    lc = [tm, mag, error, mag2, aTime, aMag, aMag2, aError, aError2]

    # only calculate these features
    # fs = feets.FeatureSpace(only=['Std', 'StetsonL'])

    fs = FeatureSpace()
    # fs = FeatureSpace(data=basicData)
    features, values = fs.extract(*lc)

    elapsed = time.time() - start
    print("Computed %s features in %.02fs" % (len(features), elapsed))

    if plot:
        g = plt.figure(2)
        plt.plot(lc[0], lc[1], "*-", alpha=0.6)
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        g.show()
        input()

    t = PrettyTable(["Feature", "Value"])
    t.align = "l"
    for i, feat in enumerate(features):
        t.add_row([feat, values[i]])

    if plot:
        print(t)

    fdict = dict(zip(features, values))

    # Ploting the example lightcurve in phase
    T = 2 * fdict["PeriodLS"]
    new_b = np.mod(lc[0], T) / T
    idx = np.argsort(2 * new_b)

    plt.plot(new_b, lc[1], '*')
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    machoTest()
