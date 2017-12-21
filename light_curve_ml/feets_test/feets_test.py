"""Testing out Feets on MACHO."""
import argparse
import logging
import time

from feets import FeatureSpace, preprocess
from feets.datasets.base import Bunch
from feets.extractors.core import DATA_TIME, DATA_MAGNITUDE, DATA_ERROR
import numpy as np
from prettytable import PrettyTable

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


#: Standard data types for EPO project
STANDARD_DATA_TYPES = ["time", "magnitude", "error"]


#: LC bands contain these time series datums
LC_DATA = (DATA_TIME, DATA_MAGNITUDE, DATA_ERROR)


#: Additional attribute for light curve Bunch data structure specifying the
#: number of bogus values removed from original data
DATA_BOGUS_REMOVED = "bogusRemoved"


#: Additional attribute for light curve Bunch data structure specifying the
#: number of statistical outliers removed from original data
DATA_OUTLIER_REMOVED = "outlierRemoved"


def parseCleanSeries(filePaths, stdLimit, errorLimit, sort=False, numBands=2,
                     limit=None):
    """Parses the given MACHO LC file into a list of noise-removed light curves
    in red and blue bands.
    :return list of Bunch"""
    lcs = []
    shortCnt = 0
    bogusCnt = 0
    outlierCnt = 0
    totalSequences = 0
    actualBands = 0
    if limit is None:
        limit = float("inf")

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
                if rBunch:
                    actualBands += 1
                else:
                    if issue == "short":
                        shortCnt += 1
                    elif issue == "bogus":
                        bogusCnt += 1
                    else:
                        outlierCnt += 1

                bBunch, issue = parseMachoBunch(series[:, 5], series[:, 8],
                                                series[:, 9], stdLimit,
                                                errorLimit)
                if bBunch:
                    actualBands += 1
                else:
                    if issue == "short":
                        shortCnt += 1
                    elif issue == "bogus":
                        bogusCnt += 1
                    else:
                        outlierCnt += 1

                lcs.append(Bunch(field=field, tile=tile, sequence=seq,
                                 category=cat, data=LC_DATA,
                                 bands=Bunch(r=rBunch, b=bBunch)))
                totalSequences += 1

            if actualBands >= limit:
                break

        if actualBands >= limit:
            break

    maxTotalBands = numBands * totalSequences
    shortRate = fmtPct(shortCnt, maxTotalBands)
    bogusRate = fmtPct(bogusCnt, maxTotalBands)
    outlierRate = fmtPct(outlierCnt, maxTotalBands)
    logger.info("Total data files: %s", totalSequences)
    logger.info("Total time series all bands: %s", maxTotalBands)
    logger.info("Failure rate all bands: short: %s bogus: %s outliers: %s",
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
    removedLcs = bandCount * lcCount - len(redBand) - len(blueBand)
    logger.info("Bands removed: %s", removedLcs)
    logger.info("Good band counts: Red: %s (%s) Blue: %s (%s)", len(redBand),
                redRate, len(blueBand), blueRate)
    logger.info("Time series length stats")
    logger.info("R-band: %s ", reportBandStats(redBand))
    logger.info("B-band: %s ", reportBandStats(blueBand))


def reportBandStats(band):
    if not band:
        return None

    lens = [len(b[DATA_TIME]) for b in band]
    bMin = min(lens)
    bMax = max(lens)
    bAve = np.average(lens)
    bStd = float(np.std(lens, dtype=np.float64))
    return "Ave: %.02f (%.02f) Min: %.02f Max: %.02f" % (bAve, bStd, bMin, bMax)


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stdLimit", type=int, default=5,
                        help="limit on the number of std's a DV value can be "
                             "from the mean beyond which the data point is "
                             "discarded")
    parser.add_argument("-e", "--errorLimit", type=int, default=3,
                        help="limit on the number of std's an error value can "
                             "be from the error mean. if exceeded, the data "
                             "point is discarded")
    parser.add_argument("-t", "--limit", type=int, default=float("inf"),
                        help="Limit script to running first n data files")
    parser.add_argument("--sort", action="store_true",
                        help="sort each light curve series by time")
    return parser.parse_args()


def _reportFeatures(features, values):
    t = PrettyTable(["Feature", "Value"])
    t.align = "l"
    for i, feat in enumerate(features):
        t.add_row([feat, values[i]])

    logger.info(t)


def machoTest():
    s = time.time()
    args = _getArgs()
    logger.info("Program args: data limit: %s stdLimit: %s errorLimit: %s",
                args.limit, args.stdLimit, args.errorLimit)

    # Load LCs
    dataDir = context_util.joinRoot("data/macho/raw")
    absPaths = sorted(absoluteFilePaths(dataDir))
    lcs = parseCleanSeries(absPaths, args.stdLimit, args.errorLimit,
                           sort=args.sort, limit=args.limit)
    lightCurveStats(lcs)
    logger.info("data load elapsed: %.2fs\n", time.time() - s)

    # compute features
    allBands = [lc.bands.r for lc in lcs if lc.bands.r]
    bBands = [lc.bands.b for lc in lcs if lc.bands.b]
    allBands.extend(bBands)  # TODO probably run separately in a function?
    if not allBands:
        logger.info("No bands")
        return

    # Top slowest features: CAR, FourierComponents, & LombScargle
    # correspond to: CAR_mean, CAR_sigma, CAR_tau,
    exclude = ["CAR_mean", "CAR_sigma", "CAR_tau"]
    fs = FeatureSpace(data=STANDARD_DATA_TYPES, exclude=exclude)
    features = fs.features_as_array_
    reportFeatures = False
    extractTimes = []
    extractLengths = []
    featureCounts = []
    for i, band in enumerate(allBands):
        if i % 30 == 0:
            logger.info("progress: %s / %s", i, len(allBands))

        # cat = band.category
        es = time.time()
        _, values = fs.extract(time=band[DATA_TIME],
                               magnitude=band[DATA_MAGNITUDE],
                               error=band[DATA_ERROR])
        extractTimes.append(time.time() - es)
        extractLengths.append(len(band[DATA_TIME]))
        featureCounts.append(len(values))
        if reportFeatures:
            _reportFeatures(features, values)

    logger.info("Ave num features: %s", np.average(featureCounts))
    totalExtractTime = np.sum(extractTimes)
    tmPer1000Points = 1000 * totalExtractTime / np.sum(extractLengths)
    logger.info("Time per 1000 data points: %.3fs", tmPer1000Points)
    logger.info("Feature extraction time: total: %.2fs mean: %.2fs min: %.2fs "
                "max: %.2fs", totalExtractTime, np.average(extractTimes),
                min(extractTimes), max(extractTimes))


if __name__ == "__main__":
    machoTest()
