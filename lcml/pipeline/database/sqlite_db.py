import numpy as np
import sqlite3

from lcml.pipeline.database.serialization import deserArray
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import joinRoot


logger = BasicLogging.getLogger(__name__)


#: CREATE TABLE for light curve time series table
CREATE_TABLE_LCS = ("CREATE TABLE IF NOT EXISTS %s ("
                    "id text primary key, "
                    "label text, "
                    "times text, "
                    "magnitudes text, "
                    "errors text)")


#: INSERT OR REPLACE light curve time series table
INSERT_REPLACE_INTO_LCS = "INSERT OR REPLACE INTO %s VALUES (?, ?, ?, ?, ?)"


CREATE_TABLE_FEATURES = ("CREATE TABLE IF NOT EXISTS %s ("
                         "id text primary key, "
                         "label text, "
                         "features text)")


INSERT_REPLACE_INTO_FEATURES = "INSERT OR REPLACE INTO %s VALUES (?, ?, ?)"


SINGLE_COL_PAGED_SELECT_QRY = ("SELECT {0} FROM {1} "
                               "WHERE {2} > {3} "
                               "ORDER BY {2} "
                               "LIMIT {4}")


def connFromParams(dbParams):
    p = joinRoot(dbParams["dbPath"])
    timeout = dbParams["timeout"]
    conn = None
    try:
        conn = sqlite3.connect(p, timeout=timeout)
    except sqlite3.OperationalError:
        logger.exception("Cannot resolve path: %s", p)

    return conn


def singleColPagingItr(cursor, table, column, selRows="*", columnInd=0,
                       pageSize=1000, textField=True):
    """Perform a find with sqlite using single-column paging to maintain a
    reasonable memory footprint.

    :param cursor: db cursor
    :param table: table to query
    :param column: single column to page over
    :param selRows: desired rows returned
    :param columnInd: 0-based index of paging column
    :param pageSize: limit on the number of records returned in a single page
    :param textField: flag specifying whether column's data type is text
    """
    prevVal = ""
    rows = True
    while rows:
        _fmtPrevVal = "\"{}\"".format(prevVal) if textField else prevVal
        q = SINGLE_COL_PAGED_SELECT_QRY.format(selRows, table, column,
                                               _fmtPrevVal, pageSize)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            yield r

        if rows:
            prevVal = rows[-1][columnInd]


SELECT_FEATURES_LABELS_QRY = "SELECT label, features FROM %s"


def selectFeaturesLabels(dbParams, limit=None):
    """Selects features and their associated labels"""
    # if this soaks up all the RAM,
    # a) try memory-mapped numpy array:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
    #
    # b) allow random subset selection:
    # choice = set(np.random.choice(setSize, subsetSize, replace=False))
    conn = connFromParams(dbParams)
    cursor = conn.cursor()

    query = SELECT_FEATURES_LABELS_QRY % dbParams["feature_table"]
    if limit:
        query += " LIMIT %s" % limit
    labels = []
    features = []
    for r in cursor.execute(query):
        rawFeats = deserArray(r[1])

        # TODO eventually remove when features have been rerun
        if not np.isfinite(rawFeats.sum()) and not np.isfinite(rawFeats).all():
            for i, f in enumerate(rawFeats):
                if not np.isfinite(f):
                    logger.warning("imputing 0.0 for: %s", f)
                    rawFeats[i] = 0.0

        features.append(rawFeats)
        labels.append(r[0])

    conn.close()
    logger.info("Feature vectors have length: %s", len(features[0]))
    return features, labels


def classLabelHistogram(dbParams):
    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    histogramQry = "SELECT label, COUNT(*) FROM %s GROUP BY label"
    cursor = cursor.execute(histogramQry % dbParams["clean_lc_table"])
    histogram = dict([_ for _ in cursor])
    conn.close()
    return histogram


def reportTableCount(cursor, table, msg=""):
    count = cursor.execute("SELECT COUNT(*) FROM %s" % table)
    logger.info("Table %s rows: %s", msg, [_ for _ in count][0][0])
