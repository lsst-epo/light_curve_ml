from typing import List, Union

import numpy as np
import sqlite3
from sqlite3 import Connection, Cursor

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


SELECT_FEATURES_LABELS_QRY = "SELECT label, features FROM %s"


def connFromParams(dbParams: dict) -> Union[Connection, None]:
    p = joinRoot(dbParams["dbPath"])
    timeout = dbParams["timeout"]
    conn = None
    try:
        conn = sqlite3.connect(p, timeout=timeout)
    except sqlite3.OperationalError:
        logger.exception("Cannot resolve path: %s", p)

    return conn


def ensureDbTables(dbParams: dict):
    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    for query, table in [(CREATE_TABLE_LCS, dbParams["raw_lc_table"]),
                         (CREATE_TABLE_LCS, dbParams["clean_lc_table"]),
                         (CREATE_TABLE_FEATURES, dbParams["feature_table"])]:
        _ensureTable(cursor, query, table)

    conn.commit()


def _ensureTable(cursor: Cursor, query: str, table: str):
    logger.info("initializing table: %s", table)
    cursor.execute(query % table)


def singleColPagingItr(cursor: Cursor, table: str, column: str,
                       selectRows: str= "*", columnInd=0,
                       pageSize: int=1000, textField: bool=True):
    """Perform a find with sqlite using single-column paging to maintain a
    reasonable memory footprint.

    :param cursor: db cursor
    :param table: table to query
    :param column: single column to page over
    :param selectRows: desired rows returned
    :param columnInd: 0-based index of paging column
    :param pageSize: limit on the number of records returned in a single page
    :param textField: flag specifying whether column's data type is text
    """
    prevVal = ""
    rows = True
    while rows:
        _fmtPrevVal = "\"{}\"".format(prevVal) if textField else prevVal
        q = SINGLE_COL_PAGED_SELECT_QRY.format(selectRows, table, column,
                                               _fmtPrevVal, pageSize)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            yield r

        if rows:
            prevVal = rows[-1][columnInd]


def selectFeaturesLabels(dbParams, limit=None) -> (List[np.ndarray], List[str]):
    """Selects light curve features and class labels"""
    # if this soaks up all the RAM,
    # a) try memory-mapped numpy array:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
    #
    # b) allow random subset selection:
    # choice = set(np.random.choice(setSize, subsetSize, replace=False))
    conn = connFromParams(dbParams)
    cursor = conn.cursor()

    query = SELECT_FEATURES_LABELS_QRY % dbParams["feature_table"]
    if limit not in (None, float("inf")):
        query += " LIMIT %s" % limit

    labels = []
    features = []
    for r in cursor.execute(query):
        features.append(deserArray(r[1]))
        labels.append(r[0])

    conn.close()
    if features:
        logger.info("Feature vectors have length: %s", len(features[0]))

    return features, labels


def classLabelHistogram(dbParams: dict) -> dict:
    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    histogramQry = "SELECT label, COUNT(*) FROM %s GROUP BY label"
    cursor = cursor.execute(histogramQry % dbParams["clean_lc_table"])
    histogram = dict([_ for _ in cursor])
    conn.close()
    return histogram


def reportTableCount(cursor: Cursor, table: str, msg: str=""):
    count = cursor.execute("SELECT COUNT(*) FROM %s" % table)
    logger.info("Table '%s' %s rows: %s", table, msg, [_ for _ in count][0][0])
