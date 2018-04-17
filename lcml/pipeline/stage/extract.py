from collections import Counter
import operator
from sqlite3 import OperationalError
from typing import List

from feets import FeatureSpace
import numpy as np
from prettytable import PrettyTable

from lcml.pipeline.database import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.database.sqlite_db import (INSERT_REPLACE_INTO_FEATURES,
                                              SINGLE_COL_PAGED_SELECT_QRY,
                                              connFromParams,
                                              reportTableCount)
from lcml.pipeline.database.serialization import deserLc, serArray
from lcml.pipeline.stage.preprocess import allFinite
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.format_util import fmtPct
from lcml.utils.multiprocess import feetsExtract, reportingImapUnordered


logger = BasicLogging.getLogger(__name__)


def feetsJobGenerator(fs, dbParams, selRows="*"):
    """Returns a generator of tuples of the form:
    (featureSpace (feets.FeatureSpace),  id (str), label (str), times (ndarray),
     mags (ndarray), errors(ndarray))
    Each tuple is used to perform a 'feets' feature extraction job.

    :param fs: feets.FeatureSpace object required to perform extraction
    :param dbParams: additional params
    :param selRows: which rows to select from clean LC table
    """
    table = dbParams["clean_lc_table"]
    pageSize = dbParams["pageSize"]
    conn = connFromParams(dbParams)
    cursor = conn.cursor()

    column = "id"  # PK
    previousId = ""  # low precedence text value
    rows = True
    while rows:
        _fmtPrevId = "\"{}\"".format(previousId)
        q = SINGLE_COL_PAGED_SELECT_QRY.format(selRows, table, column,
                                               _fmtPrevId, pageSize)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            times, mags, errors = deserLc(*r[2:])
            # intended args for lcml.utils.multiprocess._feetsExtract
            yield (fs, r[0], r[1], times, mags, errors)

        if rows:
            previousId = rows[-1][0]

    conn.close()


def feetsExtractFeatures(params: dict, dbParams: dict, limit: int):
    """Runs light curves through 'feets' library obtaining feature vectors.
    Perfoms the extraction using multiprocessing. Output order of jobs will not
    necessarily correspond to input order, therefore, class labels are returned
    with corresponding feature vectors to avoid confusion.

    :param params: extract parameters
    :param dbParams: db parameters
    :param limit: upper limit on the number of LC processed
    :returns feature vectors for each LC and list of corresponding class labels
    """
    # recommended excludes (slow): "CAR_mean", "CAR_sigma", "CAR_tau"
    # also produces nan's: "ls_fap"
    impute = params.get("impute", True)
    exclude = params["excludedFeatures"]
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    logger.info("Excluded features: %s", exclude)

    featuresTable = dbParams["feature_table"]
    ciFreq = dbParams["commitFrequency"]
    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    insertOrReplQry = INSERT_REPLACE_INTO_FEATURES % featuresTable
    reportTableCount(cursor, featuresTable, msg="before extracting")

    jobs = feetsJobGenerator(fs, dbParams)
    lcCount = 0
    skipCount = 0
    dbExceptions = 0
    imputeCounter = Counter()
    for uid, label, ftNames, features in reportingImapUnordered(feetsExtract,
                                                                jobs):
        # loop variables come from lcml.utils.multiprocess._feetsExtract
        if impute:
            _imputeFeatures(ftNames, features, imputeCounter)
        elif not allFinite(features):
            skipCount += 1
            continue

        args = (uid, label, serArray(features))
        try:
            cursor.execute(insertOrReplQry, args)
            if lcCount % ciFreq == 0:
                conn.commit()
        except OperationalError:
            logger.exception("Failed to insert %s", args)
            dbExceptions += 1

        if lcCount > limit:
            break

        lcCount += 1

    reportTableCount(cursor, featuresTable, msg="after extracting")
    conn.commit()
    conn.close()

    if skipCount:
        logger.warning("Skipped due to bad feature value rate: %s",
                       fmtPct(skipCount, lcCount))

    if dbExceptions:
        logger.warning("Db exception count: %s", dbExceptions)

    if imputeCounter:
        t = PrettyTable(["feature name", "imputes", "impute rate",
                         "percentage of all imputes"])
        totalImputes = sum(imputeCounter.values())
        for name, count in sorted(imputeCounter.items(),
                                  key=operator.itemgetter(1), reverse=True):
            t.add_row([name, count, fmtPct(count, lcCount),
                       fmtPct(count, totalImputes)])

        logger.info("\n" + str(t))


def _imputeFeatures(featureNames: List[str], featureValues: List[float],
                    imputes: Counter):
    """Sets non-finite feature values to 0.0"""
    for i, v in enumerate(featureValues):
        if not np.isfinite(v):
            logger.warning("imputing: %s %s => 0.0", featureNames[i],
                           featureValues[i])
            imputes[featureNames[i]] += 1
            featureValues[i] = 0.0
