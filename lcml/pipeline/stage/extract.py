from sqlite3 import OperationalError

from feets import FeatureSpace


from lcml.pipeline.database import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.database.serialization import deserLc, serArray
from lcml.pipeline.database.sqlite_db import (INSERT_REPLACE_INTO_FEATURES,
                                              SINGLE_COL_PAGED_SELECT_QRY,
                                              connFromParams,
                                              reportTableCount)
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.multiprocess import feetsExtract, reportingImapUnordered


logger = BasicLogging.getLogger(__name__)


def feetsJobGenerator(fs: FeatureSpace, dbParams: dict, tableName: str,
                      selRows: str="*", offset: int=0):
    """Returns a generator of tuples of the form:
    (featureSpace (feets.FeatureSpace),  id (str), label (str), times (ndarray),
     mags (ndarray), errors(ndarray))
    Each tuple is used to perform a 'feets' feature extraction job.

    :param fs: feets.FeatureSpace object required to perform extraction
    :param dbParams: additional params
    :param tableName: table containing light curves
    :param selRows: which rows to select from clean LC table
    :param offset: number of light curves to skip in db table before processing
    """
    pageSize = dbParams["pageSize"]
    conn = connFromParams(dbParams)
    cursor = conn.cursor()

    column = "id"  # PK
    previousId = ""  # low precedence text value
    rows = True
    while rows:
        _fmtPrevId = "\"{}\"".format(previousId)
        q = SINGLE_COL_PAGED_SELECT_QRY.format(selRows, tableName, column,
                                               _fmtPrevId, pageSize, offset)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            times, mags, errors = deserLc(*r[2:])
            # intended args for lcml.utils.multiprocess._feetsExtract
            yield (fs, r[0], r[1], times, mags, errors)

        if rows:
            previousId = rows[-1][0]

    conn.close()


def feetsExtractFeatures(extractParams: dict, dbParams: dict, lcTable: str,
                         featuresTable: str, limit: int):
    """Runs light curves through 'feets' library obtaining feature vectors.
    Perfoms the extraction using multiprocessing. Output order of jobs will not
    necessarily correspond to input order, therefore, class labels are returned
    with corresponding feature vectors to avoid confusion.

    :param extractParams: extract parameters
    :param dbParams: db parameters
    :param lcTable: name of lc table
    :param featuresTable: name of features table
    :param limit: upper limit on the number of LC processed
    :returns feature vectors for each LC and list of corresponding class labels
    """
    # recommended excludes (slow): "CAR_mean", "CAR_sigma", "CAR_tau"
    # also produces nan's: "ls_fap"
    exclude = extractParams["excludedFeatures"]
    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    logger.info("Excluded features: %s", exclude)

    ciFreq = dbParams["commitFrequency"]
    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    insertOrReplQry = INSERT_REPLACE_INTO_FEATURES % featuresTable
    reportTableCount(cursor, featuresTable, msg="before extracting")

    offset = extractParams.get("offset", 0)
    logger.info("Beginning extraction at offset: %s in LC table", offset)

    jobs = feetsJobGenerator(fs, dbParams, lcTable, offset=offset)
    lcCount = 0
    dbExceptions = 0
    for uid, label, ftNames, features in reportingImapUnordered(feetsExtract,
                                                                jobs):
        # loop variables come from lcml.utils.multiprocess._feetsExtract
        args = (uid, label, serArray(features))
        try:
            cursor.execute(insertOrReplQry, args)
            if lcCount % ciFreq == 0:
                logger.info("commit progress: %s", lcCount)
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

    if dbExceptions:
        logger.warning("Db exception count: %s", dbExceptions)
