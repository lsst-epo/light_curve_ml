import cPickle

import numpy as np

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


#: CREATE TABLE for light curve time series table
LC_TABLE_CREATE_QRY = ("CREATE TABLE IF NOT EXISTS %s ("
                       "id text primary key, "
                       "label text, "
                       "times text, "
                       "magnitudes text, "
                       "errors text)")


#: INSERT OR REPLACE light curve time series table
LC_TABLE_INSERT_QRY = "INSERT OR REPLACE INTO {} VALUES (?, ?, ?, ?, ?)"


def serLc(times, mags, errors):
    """Serializes light curve as 3 arrays or lists to 3 binary strings"""
    t = serArray(times)
    m = serArray(mags)
    e = serArray(errors)
    return t, m, e


def serArray(a):
    return cPickle.dumps(a)


def deserLc(times, mags, errors):
    """Deserializes 3 binary string light curves to 3 numpy arrays of float64
    (equivalent to python float)."""
    t = deserArray(times)
    m = deserArray(mags)
    e = deserArray(errors)
    return t, m, e


def deserArray(binStr):
    return np.array(cPickle.loads(str(binStr)), dtype=np.float64)


def reportTableCount(cursor, table, msg=""):
    count = cursor.execute("SELECT COUNT(*) FROM %s" % table)
    logger.info("Table %s rows: %s", msg, [_ for _ in count][0][0])