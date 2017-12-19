"""Testing out Feets on MACHO."""
import argparse
import logging
import time

from feets import preprocess
from feets.datasets.base import Bunch
from feets.extractors.core import DATA_TIME, DATA_MAGNITUDE, DATA_ERROR
import numpy as np

from light_curve_ml.utils import context_util
from light_curve_ml.utils.basic_logging import getBasicLogger
from light_curve_ml.utils.context_util import absoluteFilePaths
from light_curve_ml.utils.data_util import (removeMachoOutliers,
                                            SUFFICIENT_LC_DATA)
from light_curve_ml.utils.format_util import fmtPct

logger = getBasicLogger(__name__, __file__)


#: all data types accepted by the library
ALL_DATA_TYPES = ["time", "magnitude", "error", "magnitude2", "aligned_time",
                  "aligned_magnitude", "aligned_error", "aligned_magnitude2",
                  "aligned_error2"]


#: LC bands contain these time series datums
LC_DATA = (DATA_TIME, DATA_MAGNITUDE, DATA_ERROR)


#: Additional attribute for light curve Bunch data structure specifying the
#: number of bogus values removed from original data
DATA_BOGUS_REMOVED = "bogusRemoved"


#: Additional attribute for light curve Bunch data structure specifying the
#: number of statistical outliers removed from original data
DATA_OUTLIER_REMOVED = "outlierRemoved"


def parseCleanSeries(filePaths, stdLimit, errorLimit, sort=False, numBands=2):
    """Parses the given MACHO LC file into a list of noise-removed light curves
    in red and blue bands.
    :return list of Bunch"""
    lcs = []
    rSuccess = 0
    bSuccess = 0
    shortCnt = 0
    bogusCnt = 0
    outlierCnt = 0
    totalSequences = 0
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
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("field: %s tile: %s seqn: %s", field, tile,
                                 seq)

                series = tileData[np.where(tileData[:, 3] == seq)]
                cat = series[0, 0]
                if sort:
                    # sort by mjd time to be sure
                    series = series[series[:, 5].argsort()]

                rBunch, issue = parseMachoBunch(series[:, 5], series[:, 6],
                                                series[:, 7], stdLimit,
                                                errorLimit)
                if issue is None:
                    rSuccess += 1
                elif issue == "short":
                    shortCnt += 1
                elif issue == "bogus":
                    bogusCnt += 1
                else:
                    outlierCnt += 1

                bBunch, issue = parseMachoBunch(series[:, 5], series[:, 8],
                                                series[:, 9], stdLimit,
                                                errorLimit)
                if issue is None:
                    bSuccess += 1
                elif issue == "short":
                    shortCnt += 1
                elif issue == "bogus":
                    bogusCnt += 1
                else:
                    outlierCnt += 1

                lcs.append(Bunch(field=field, tile=tile, sequence=seq,
                                 category=cat, data=LC_DATA,
                                 bands=Bunch(r=rBunch, b=bBunch)))
                totalSequences += 1

    redRate = fmtPct(rSuccess, totalSequences)
    blueRate = fmtPct(bSuccess, totalSequences)

    totalBands = numBands * totalSequences
    shortRate = fmtPct(shortCnt, totalBands)
    bogusRate = fmtPct(bogusCnt, totalBands)
    outlierRate = fmtPct(outlierCnt, totalBands)
    logger.info("Total data files: %s bands: %s", totalSequences, totalBands)
    logger.info("Success rate: R-band: %s B-band: %s", redRate, blueRate)
    logger.info("All bands failure rate: short: %s bogus: %s outliers: %s",
                shortRate, bogusRate, outlierRate)
    return lcs


def parseMachoBunch(timeData, magData, errorData, stdLimit, errorLimit):
    if len(timeData) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s to start", len(timeData))
        return None, "short"

    # removes -99's endemic to MACHO
    tm, mag, err = removeMachoOutliers(timeData, magData, errorData,
                                       remove=-99.0)
    bogusRemoved = len(timeData) - len(tm)
    if bogusRemoved:
        logger.debug("bogus removed %s", bogusRemoved)

    if len(tm) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s after removing -99", len(tm))
        return None, "bogus"

    # removes statistical outliers
    _tm, _mag, _err = preprocess.remove_noise(tm, mag, err,
                                              error_limit=errorLimit,
                                              std_limit=stdLimit)
    outlierRemoved = len(tm) - len(_tm)
    if outlierRemoved:
        logger.debug("outlier removed %s", outlierRemoved)

    if len(_tm) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient length: %s after statistical outliers "
                       "removed", len(_tm))
        return None, "outliers"

    b = Bunch(**{DATA_TIME: _tm, DATA_MAGNITUDE: _mag, DATA_ERROR: _err})
    b[DATA_BOGUS_REMOVED] = bogusRemoved
    b[DATA_OUTLIER_REMOVED] = outlierRemoved
    return b, None


def testStats():
    b1 = np.zeros(200)
    b2 = np.zeros(60)
    rBunch = Bunch(**{DATA_TIME: b1, DATA_MAGNITUDE: b1, DATA_ERROR: b1})
    bBunch = Bunch(**{DATA_TIME: b2, DATA_MAGNITUDE: b2, DATA_ERROR: b2})
    bnc = Bunch(bands=Bunch(r=rBunch, b=bBunch))
    lightCurveStats([bnc])


def lightCurveStats(lcs):
    """Computes basic stats on light curves"""
    bandCount = len(lcs[0].bands)
    lcCount = len(lcs)

    redBand = [lc.bands.r for lc in lcs if lc.bands.r]
    redRate = fmtPct(len(redBand), lcCount)

    blueBand = [lc.bands.b for lc in lcs if lc.bands.b]
    blueRate = fmtPct(len(blueBand), lcCount)
    removedLcs = 2 * lcCount - len(redBand) - len(blueBand)
    logger.info("LC report")
    logger.info("Count: %s (bands: %s) Removed: %s Appearance rate: Red: %s"
                " Blue: %s", lcCount, bandCount, removedLcs, redRate, blueRate)
    logger.info("LC length stats")
    logger.info("R-band: %s ", reportBandStats(redBand))
    logger.info("B-band: %s ", reportBandStats(blueBand))


def reportBandStats(band):
    lens = [len(b[DATA_TIME]) for b in band]
    bMin = min(lens)
    bMax = max(lens)
    bAve = np.average(lens)
    bStd = float(np.std(lens, dtype=np.float64))
    return "LCs: %s Ave: %.02f (%.02f) Min: %.02f Max: %.02f" % (
        len(band), bAve, bStd, bMin, bMax)


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stdLimit", type=int,
                        help="limit on the distance the DV value can be from "
                             "the mean beyond which the data point is "
                             "discarded")
    parser.add_argument("-e", "--errorLimit", type=int,
                        help="limit on the distance an error value can be from"
                             "the error mean beyond which the data point is"
                             "discarded")
    parser.add_argument("-s", "--sort", type=bool, action="store_true",
                        help="sort each light curve series by time")
    return parser.parse_args()


def machoTest():
    s = time.time()
    args = _getArgs()

    # std_limit = 5, error_limit = 3  # Feets defaults
    # stdLimit = 5
    # errorLimit = 3
    # sortSeries = False
    logger.info("Parameters: stdLimit: %s errorLimit: %s", args.stdLimit,
                args.errorLimit)
    dataDir = context_util.joinRoot("data/macho/raw")
    absPaths = absoluteFilePaths(dataDir)

    lcs = parseCleanSeries(absPaths, args.stdLimit, args.errorLimit,
                           sort=args.sortSeries)
    lightCurveStats(lcs)
    logger.info("elapsed: %.2fs", time.time() - s)

    # next...
    # bands = [lc.bands.r for lc in lcs if lc.bands.r]
    # bBands = [lc.bands.b for lc in lcs if lc.bands.b]
    # bands.extend(bBands)
    # for band in bands:
    #     lc = [band[DATA_TIME], band[DATA_MAGNITUDE], band[DATA_ERROR]]
    #
    #     fs = FeatureSpace()
    #     # fs = FeatureSpace(data=basicData)
    #     features, values = fs.extract(*lc)
    #
    #     t = PrettyTable(["Feature", "Value"])
    #     t.align = "l"
    #     for i, feat in enumerate(features):
    #         t.add_row([feat, values[i]])


if __name__ == "__main__":
    machoTest()
    # testStats()  # to do unit test
