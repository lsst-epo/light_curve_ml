import cPickle
import sqlite3

import numpy as np

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


SINGLE_COL_PAGED_SELECT_QRY = ("SELECT * FROM {0} "
                               "WHERE {1} > \"{2}\" "
                               "ORDER BY {1} "
                               "LIMIT {3}")


def connFromParams(dbParams):
    return sqlite3.connect(joinRoot(dbParams["dbPath"]))


def serLc(times, mags, errors):
    """Serializes light curve attributes (as arrays or lists) to binary strings
    """
    t = serArray(times)
    m = serArray(mags)
    e = serArray(errors)
    return t, m, e


def serArray(a):
    return cPickle.dumps(a)


def deserLc(times, mags, errors):
    """Deserializes 3 binary-string light curves to 3 numpy arrays of float64
    (equivalent of Python float)."""
    t = deserArray(times)
    m = deserArray(mags)
    e = deserArray(errors)
    return t, m, e


def deserArray(binStr):
    return np.array(cPickle.loads(str(binStr)), dtype=np.float64)


def reportTableCount(cursor, table, msg=""):
    count = cursor.execute("SELECT COUNT(*) FROM %s" % table)
    logger.info("Table %s rows: %s", msg, [_ for _ in count][0][0])


def singleColPagingItr(cursor, table, column, columnInd=0, pageSize=1000):
    """Perform a find with sqlite using single-column paging to maintain a
    reasonable memory footprint.
    """
    # TODO remove current assumptions: column value is text, all rows selected
    previousValue = ""
    rows = True
    while rows:
        # assumes column is type text, remove that
        q = SINGLE_COL_PAGED_SELECT_QRY.format(table, column, previousValue,
                                               pageSize)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            yield r

        if rows:
            previousValue = rows[-1][columnInd]