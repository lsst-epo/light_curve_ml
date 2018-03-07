import numpy as np
import sqlite3

from feets import preprocess

from lcml.pipeline.data_format.db_schema import (LC_TABLE_CREATE_QRY,
                                                 LC_TABLE_INSERT_QRY,
                                                 reportTableCount, deserLc,
                                                 serLc)
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import joinRoot
from lcml.utils.format_util import fmtPct


logger = BasicLogging.getLogger(__name__)


#: Additional attribute for light curve Bunch data structure specifying the
#: number of bogus values removed from original data
DATA_BOGUS_REMOVED = "bogusRemoved"


#: Additional attribute for light curve Bunch data structure specifying the
#: number of statistical outliers removed from original data
DATA_OUTLIER_REMOVED = "outlierRemoved"


#: cannot use LC because there is simply not enough data to go on
INSUFFICIENT_DATA_REASON = "insufficient at start"


#: cannot use LC because there is insufficient data after removing bogus values
BOGUS_DATA_REASON = "insufficient due to bogus data"


#: cannot use LC because there is insufficient data after removing statistical
#: outliers
OUTLIERS_REASON = "insufficient due to statistical outliers"


#: Research by Kim suggests it best that light curves have at least 80 data
#: points for accurate classification
SUFFICIENT_LC_DATA = 80


#: data values to scrub; nb np.nan != float("nan") but np.inf == float("inf")
NON_FINITE_VALUES = {np.nan, float("nan"), float("inf"), float("-inf")}


def preprocessLc(timeData, magData, errorData, removes, stdLimit, errorLimit):
    """Returns a cleaned version of an LC. LC may be deemed unfit for use, in
    which case the reason for rejection is specified.

    :returns processed lc as a tuple and failure reason (string)
    """
    removedCounts = {DATA_BOGUS_REMOVED: 0, DATA_OUTLIER_REMOVED: 0}
    if len(timeData) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s to start", len(timeData))
        return None, INSUFFICIENT_DATA_REASON, removedCounts

    # remove bogus data
    tm, mag, err = lcFilterBogus(timeData, magData, errorData, removes=removes)
    removedCounts[DATA_BOGUS_REMOVED] = len(timeData) - len(tm)
    if len(tm) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s after removing bogus values", len(tm))
        return None, BOGUS_DATA_REASON, removedCounts

    # removes statistical outliers
    _tm, _mag, _err = preprocess.remove_noise(tm, mag, err,
                                              error_limit=errorLimit,
                                              std_limit=stdLimit)
    removedCounts[DATA_OUTLIER_REMOVED] = len(tm) - len(_tm)
    if len(_tm) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s after statistical outliers removed",
                     len(_tm))
        return None, OUTLIERS_REASON, removedCounts

    return (_tm, _mag, _err), None, removedCounts


#: Default value for preprocessing param `std threshold`
DEFAULT_STD_LIMIT = 5


#: Default value for preprocessing param `error limit`
DEFAULT_ERROR_LIMIT = 3


def cleanLightCurves(params, dbParams):
    """Clean lightcurves and report details on discards."""
    removes = set(params["filter"]) if "filter" in params else set()
    removes = removes.union(NON_FINITE_VALUES)
    stdLimit = params.get("stdLimit", DEFAULT_STD_LIMIT)
    errorLimit = params.get("errorLimit", DEFAULT_ERROR_LIMIT)

    rawTable = dbParams["raw_lc_table"]
    cleanTable = dbParams["clean_lc_table"]
    commitFrequency = dbParams["commitFrequency"]
    conn = sqlite3.connect(joinRoot(dbParams["dbPath"]))
    cursor = conn.cursor()
    cursor.execute(LC_TABLE_CREATE_QRY % cleanTable)
    reportTableCount(cursor, cleanTable, msg="before cleaning")
    insertOrReplace = LC_TABLE_INSERT_QRY.format(cleanTable)

    shortIssueCount = 0
    bogusIssueCount = 0
    outlierIssueCount = 0

    totalLcs = cursor.execute("SELECT COUNT(*) from %s" % rawTable).next()[0]

    # results = [_ for _ in cursor.execute("SELECT * FROM %s" % rawTable)]
    results = cursor.execute("SELECT * FROM %s" % rawTable)
    insertCount = 0
    for r in results:
        times, mags, errors = deserLc(*r[2:])
        lc, issue, _ = preprocessLc(times, mags, errors, removes=removes,
                                    stdLimit=stdLimit, errorLimit=errorLimit)
        if lc:
            args = (r[0], r[1]) + serLc(*lc)
            cursor.execute(insertOrReplace, args)
            insertCount += 1
            if insertCount % commitFrequency == 0:
                logger.critical("progress: %s", insertCount)
                conn.commit()

        elif issue == INSUFFICIENT_DATA_REASON:
            shortIssueCount += 1
        elif issue == BOGUS_DATA_REASON:
            bogusIssueCount += 1
        elif issue == OUTLIERS_REASON:
            outlierIssueCount += 1
        else:
            raise ValueError("Bad reason: %s" % issue)

    reportTableCount(cursor, cleanTable, msg="after cleaning")
    conn.commit()
    conn.close()

    passRate = fmtPct(insertCount, totalLcs)
    shortRate = fmtPct(shortIssueCount, totalLcs)
    bogusRate = fmtPct(bogusIssueCount, totalLcs)
    outlierRate = fmtPct(outlierIssueCount, totalLcs)
    logger.info("Dataset size: %d Pass rate: %s", totalLcs, passRate)
    logger.info("Discard rates: short: %s bogus: %s outlier: %s", shortRate,
                bogusRate, outlierRate)


def lcFilterBogus(mjds, values, errors, removes):
    """Simple light curve filter that removes bogus magnitude and error
    values."""
    return zip(*[(mjds[i], v, errors[i])
                 for i, v in enumerate(values)
                 if v not in removes and errors[i] not in removes])


def allFinite(X):
    """Adapted from sklearn.utils.validation._assert_all_finite"""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.

    # TODO consider verifying float dtype all same??
    return (False
            if X.dtype.char in np.typecodes['AllFloat'] and
               not np.isfinite(X.sum()) and not np.isfinite(X).all()
            else True)
