import cPickle
import numpy as np
import sqlite3

from feets import preprocess
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


def cleanLightCurves(params, stdLimit=5, errorLimit=3):
    """Clean lightcurves and report details on discards"""
    removes = set(params["filter"]) if "filter" in params else set()
    removes = removes.union(NON_FINITE_VALUES)
    batchSize = params["batchSize"]

    rawTable = params["raw_lc_table"]
    cleanTable = params["clean_lc_table"]
    conn = sqlite3.connect(joinRoot(params["dbPath"]))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS %s (id text primary key, "
                   "label text, times text, magnitudes text, errors text)" %
                   cleanTable)
    insertOrReplace = "INSERT OR REPLACE INTO {} VALUES (?, ?, ?, ?, ?)".format(
        cleanTable)

    shortIssueCount = 0
    bogusIssueCount = 0
    outlierIssueCount = 0

    totalLcs = cursor.execute("SELECT COUNT(*) from %s" % rawTable).next()[0]
    results = [_ for _ in cursor.execute("SELECT * FROM %s" % rawTable)]
    insertCount = 0
    for row in results:
        times = np.array(cPickle.loads(str(row[2])), dtype=np.float32)
        mags = np.array(cPickle.loads(str(row[3])), dtype=np.float32)
        errors = np.array(cPickle.loads(str(row[4])), dtype=np.float32)
        lc, issue, _ = preprocessLc(times, mags, errors, removes=removes,
                                    stdLimit=stdLimit, errorLimit=errorLimit)
        if lc:
            args = (row[0], row[1], cPickle.dumps(lc[0]), cPickle.dumps(lc[1]),
                    cPickle.dumps(lc[2]))
            cursor.execute(insertOrReplace, args)
            insertCount += 1
            if not insertCount % batchSize:
                # fix logging config
                logger.critical("progress: %s", insertCount)
                conn.commit()

        else:
            if issue == INSUFFICIENT_DATA_REASON:
                shortIssueCount += 1
            elif issue == BOGUS_DATA_REASON:
                bogusIssueCount += 1
            elif issue == OUTLIERS_REASON:
                outlierIssueCount += 1
            else:
                raise ValueError("Bad reason: %s" % issue)

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
    return (False
            if X.dtype.char in np.typecodes['AllFloat'] and
               not np.isfinite(X.sum()) and not np.isfinite(X).all()
            else True)