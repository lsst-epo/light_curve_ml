import sqlite3

from feets import FeatureSpace

from lcml.pipeline.data_format import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.data_format.db_schema import serArray, deserLc
from lcml.pipeline.preprocess import allFinite, NON_FINITE_VALUES
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import joinRoot
from lcml.utils.format_util import fmtPct
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess


logger = BasicLogging.getLogger(__name__)


def _workGenerator(cursor, fs, dbParams):
    table = dbParams["clean_lc_table"]
    column = "id"
    lastValue = ""
    pageSize = 1000

    qBase = ("SELECT * FROM {0} "
             "WHERE {1} > \"{2}\" "
             "ORDER BY {1} "
             "LIMIT {3}")

    rows = True
    while rows:
        q = qBase.format(table, column, lastValue, pageSize)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            times, mags, errors = deserLc(*r[2:])
            # TODO may be useful to include the OGLE3_ID so that it may included in the feature table
            yield (fs, r[1], times, mags, errors)

        if rows:
            lastValue = rows[-1][0]


def feetsExtractFeatures(params, dbParams):
    """Runs light curves through 'feets' library obtaining feature vectors.
    Perfoms the extraction using multiprocessing. Output order will not
    necessarily correspond to input order, therefore, class labels are returned
    as well aligned with feature vectors to avoid confusion.

    :param params: extract parameters
    :param dbParams: db parameters
    :returns feature vectors for each LC and list of corresponding class labels
    """
    # recommended excludes (slow): "CAR_mean", "CAR_sigma", "CAR_tau"
    # also produces nan's: "ls_fap"
    exclude = params["excludedFeatures"]
    logger.info("Excluded features: %s", exclude)
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)

    impute = params.get("impute", True)
    badCount = 0
    totalCount = 0
    table = dbParams["feature_table"]

    conn = sqlite3.connect(joinRoot(dbParams["dbPath"]))
    cursor = conn.cursor()
    extractJobs = _workGenerator(cursor, fs, dbParams)

    # TODO create features table if not exists command
    # UID, CLASS_LABEL, FEATURES

    # TODO insert or replace query
    FEATURES_TABLE_INSERT_QRY = "INSERT OR REPLACE INTO {} VALUES (?, ?)"
    insertOrReplace = FEATURES_TABLE_INSERT_QRY.format(table)
    for features, label in mapMultiprocess(feetsExtract, extractJobs):
        if allFinite(features):
            _writeFeatures(cursor, insertOrReplace, label, features)
        elif impute:
            for i, f in enumerate(features):
                if f in NON_FINITE_VALUES:
                    # set non-finite feature values to 0.0
                    logger.warning("imputing feature: %s value: %s", i, f)
                    features[i] = 0.0

            _writeFeatures(cursor, insertOrReplace, label, features)
            # TODO work out the 2x2 possiblity matrix and make a single writeFeatures call
        else:
            badCount += 1

        totalCount += 1

        # TODO commit at some frequency

    if badCount:
        logger.warning("Bad feature value rate: %s", fmtPct(badCount,
                                                            totalCount))


def _writeFeatures(cursor, query, label, features):
    pass
    # args = (label, serArray(features))
    # cursor.execute(query, args)
