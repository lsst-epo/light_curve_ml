"""Testing out Feets on MACHO."""
import argparse
import logging
import multiprocessing
from multiprocessing import Pool
import time

from feets import FeatureSpace
from feets.datasets.base import Bunch
from feets.extractors.core import DATA_TIME, DATA_MAGNITUDE, DATA_ERROR
import numpy as np
from prettytable import PrettyTable

from lcml.pipeline.data_format import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.preprocess import preprocessLc
from lcml.utils import context_util
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import absoluteFilePaths
from lcml.utils.format_util import fmtPct


logger = BasicLogging.getLogger(__name__)


#: all data types accepted by the library
ALL_DATA_TYPES = ["time", "magnitude", "error", "magnitude2", "aligned_time",
                  "aligned_magnitude", "aligned_error", "aligned_magnitude2",
                  "aligned_error2"]


#: LC bands contain these time series datums
LC_DATA = (DATA_TIME, DATA_MAGNITUDE, DATA_ERROR)


_GARBAGE_VALUES = {float("nan"), float("-inf"), float("inf")}
_MACHO_REMOVE = _GARBAGE_VALUES.union({-99.0})


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

                rLc, issue, _ = preprocessLc(series[:, 5], series[:, 6],
                                             series[:, 7], _MACHO_REMOVE,
                                             stdLimit, errorLimit)
                if rLc:
                    actualBands += 1
                else:
                    if issue == "short":
                        shortCnt += 1
                    elif issue == "bogus":
                        bogusCnt += 1
                    else:
                        outlierCnt += 1

                bLc, issue, _ = preprocessLc(series[:, 5], series[:, 8],
                                             series[:, 9], _MACHO_REMOVE,
                                             stdLimit, errorLimit)
                if bLc:
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
                                 bands=Bunch(r=rLc, b=bLc)))
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

    lens = [len(b[0]) for b in band]
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
    parser.add_argument("--allFeatures", action="store_true", help="If "
                        "specified, all features will be extracted, otherwise,"
                        " some slow features will be omitted.")
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
    absPaths = sorted(absoluteFilePaths(dataDir, ext="csv"))
    lcs = parseCleanSeries(absPaths, args.stdLimit, args.errorLimit,
                           sort=args.sort, limit=args.limit)
    lightCurveStats(lcs)
    logger.info("data load elapsed: %.2fs\n", time.time() - s)

    # compute features
    allBands = [(lc.category, lc.bands.r) for lc in lcs if lc.bands.r]
    bBands = [(lc.category, lc.bands.b) for lc in lcs if lc.bands.b]
    allBands.extend(bBands)
    if not allBands:
        logger.info("No bands")
        return

    # Top slowest features: CAR, FourierComponents, & LombScargle
    # correspond to: CAR_mean, CAR_sigma, CAR_tau,
    exclude = [] if args.allFeatures else ["CAR_mean", "CAR_sigma", "CAR_tau"]
    logger.info("Excluded features: %s", exclude)
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    features = fs.features_as_array_
    reportFeatures = False
    extractTimes = []
    extractLengths = []
    featureCounts = []

    pool = Pool(processes=multiprocessing.cpu_count())
    extractStart = time.time()
    categoryValuesLengthRes = pool.map(extractWork, [(fs, cat, b)
                                                     for cat, b in allBands])
    extractElapsedMin = (time.time() - extractStart) / 60
    for category, values, length, elapsed in categoryValuesLengthRes:

        featureCounts.append(len(values))
        extractLengths.append(length)
        extractTimes.append(elapsed)

        if reportFeatures:
            _reportFeatures(features, values)

    logger.info("Extract elapsed: %.2fm", extractElapsedMin)
    logger.info("Ave num features: %s", np.average(featureCounts))
    totalExtractTime = np.sum(extractTimes)
    tmPer1000Points = 1000 * totalExtractTime / np.sum(extractLengths)
    logger.info("Time per 1000 data points: %.3fs", tmPer1000Points)
    logger.info("Feature extraction time: total: %.2fs mean: %.2fs min: %.2fs "
                "max: %.2fs", totalExtractTime, np.average(extractTimes),
                min(extractTimes), max(extractTimes))


def extractWork(args):
    """Accepts a FeatureSpace, category, and LC band including time, magnitude,
     and error. Returns the category, the feature values, LC length, and
     feature extraction time."""
    featureSpace = args[0]
    category = args[1]
    band = args[2]
    startTime = time.time()
    _, values = featureSpace.extract(time=band[0],
                                     magnitude=band[1],
                                     error=band[2])
    elapsedTime = time.time() - startTime
    return category, values, len(band[0]), elapsedTime


if __name__ == "__main__":
    machoTest()
