from feets import preprocess
from feets.extractors.core import DATA_TIME, DATA_MAGNITUDE, DATA_ERROR
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.data_util import SUFFICIENT_LC_DATA, lcFilterBogus


logger = getBasicLogger(__name__, __file__)


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


def preprocessLc(timeData, magData, errorData, remove=(-99.0,), stdLimit=5,
                 errorLimit=3):
    """Returns a cleaned version of an LC. LC may be deemed unfit for use, in
    which case the reason for rejection is specified.

    :returns processed lc as a tuple and failure reason (string)
    """
    if len(timeData) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s to start", len(timeData))
        return None, INSUFFICIENT_DATA_REASON

    # remove bogus data
    tm, mag, err = lcFilterBogus(timeData, magData, errorData, remove=remove)
    bogusRemoved = len(timeData) - len(tm)
    if bogusRemoved:
        logger.debug("bogus removed %s", bogusRemoved)

    if len(tm) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s after removing bogus values", len(tm))
        return None, BOGUS_DATA_REASON

    # removes statistical outliers
    _tm, _mag, _err = preprocess.remove_noise(tm, mag, err,
                                              error_limit=errorLimit,
                                              std_limit=stdLimit)
    outlierRemoved = len(tm) - len(_tm)
    if outlierRemoved:
        logger.debug("outlier removed %s", outlierRemoved)

    if len(_tm) < SUFFICIENT_LC_DATA:
        logger.debug("insufficient: %s after statistical outliers removed",
                     len(_tm))
        return None, OUTLIERS_REASON

    lc = {DATA_TIME: _tm, DATA_MAGNITUDE: _mag, DATA_ERROR: _err,
          DATA_BOGUS_REMOVED: bogusRemoved,
          DATA_OUTLIER_REMOVED: outlierRemoved}
    return lc, None
