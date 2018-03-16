import sqlite3

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
    p = joinRoot(dbParams["dbPath"])
    timeout = dbParams["timeout"]
    conn = None
    try:
        conn = sqlite3.connect(p, timeout=timeout)
    except sqlite3.OperationalError:
        logger.exception("Cannot resolve path: %s", p)

    return conn


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
