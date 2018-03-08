from feets import FeatureSpace

from lcml.pipeline.data_format import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.data_format.db_format import (CREATE_TABLE_FEATURES,
                                                 INSERT_REPLACE_INTO_FEATURES,
                                                 SINGLE_COL_PAGED_SELECT_QRY,
                                                 connFromParams, deserLc,
                                                 reportTableCount, serArray)
from lcml.pipeline.preprocess import allFinite, NON_FINITE_VALUES
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.format_util import fmtPct
from lcml.utils.multiprocess import feetsExtract, multiprocessMapGenerator


logger = BasicLogging.getLogger(__name__)


def feetsJobGenerator(fs, dbParams):
    """Returns a generator of tuples of the form:
    (featureSpace (feets.FeatureSpace),  id (str), label (str), times (ndarray),
     mags (ndarray), errors(ndarray))
    Each tuple is used to perform a 'feets' feature extraction job.

    :param fs: feets.FeatureSpace object required to perform extraction
    :param dbParams: additional params
    """
    table = dbParams["clean_lc_table"]
    pageSize = dbParams["pageSize"]
    conn = connFromParams(dbParams)
    cursor = conn.cursor()

    column = "id"  # PK
    previousId = ""  # low precedence text value
    rows = True
    while rows:
        q = SINGLE_COL_PAGED_SELECT_QRY.format(table, column, previousId,
                                               pageSize)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            times, mags, errors = deserLc(*r[2:])
            # intended args for lcml.utils.multiprocess._feetsExtract
            yield (fs, r[0], r[1], times, mags, errors)

        if rows:
            previousId = rows[-1][0]

    conn.close()


def feetsExtractFeatures(params, dbParams):
    """Runs light curves through 'feets' library obtaining feature vectors.
    Perfoms the extraction using multiprocessing. Output order of jobs will not
    necessarily correspond to input order, therefore, class labels are returned
    with corresponding feature vectors to avoid confusion.

    :param params: extract parameters
    :param dbParams: db parameters
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
    cursor.execute(CREATE_TABLE_FEATURES % featuresTable)
    insertOrReplQry = INSERT_REPLACE_INTO_FEATURES % featuresTable
    reportTableCount(cursor, featuresTable, msg="before extracting")

    jobs = feetsJobGenerator(fs, dbParams)
    skippedLcCount = 0
    totalLcCount = 0
    for uid, label, ftNames, features in multiprocessMapGenerator(feetsExtract,
                                                                  jobs):
        # loop variables come from lcml.utils.multiprocess._feetsExtract
        totalLcCount += 1
        if impute:
            _imputeFeatures(ftNames, features)
        elif not allFinite(features):
            skippedLcCount += 1
            continue

        args = (uid, label, serArray(features))
        cursor.execute(insertOrReplQry, args)
        if totalLcCount % ciFreq == 0:
            conn.commit()

    reportTableCount(cursor, featuresTable, msg="after extracting")
    conn.commit()
    conn.close()
    if skippedLcCount:
        logger.warning("Skipped due to bad feature value rate: %s",
                       fmtPct(skippedLcCount, totalLcCount))


def _imputeFeatures(featureNames, featureValues):
    """Sets non-finite feature values to 0.0"""
    for i, v in enumerate(featureValues):
        if v in NON_FINITE_VALUES:
            logger.warning("imputing: %s %s => 0.0", featureNames[i],
                           featureValues[i])
            featureValues[i] = 0.0
