"""These functions are duck-typed for input: dataDir: str, limit: int and
outputs: labels: List[str], times: List[ndarray], magnitudes: List[ndarray],
errors: List[ndarray]"""
from abc import abstractmethod
import csv
import logging

from lcml.pipeline.database.sqlite_db import (CREATE_TABLE_LCS,
                                              INSERT_REPLACE_INTO_LCS,
                                              connFromParams,
                                              reportTableCount)
from lcml.pipeline.database.serialization import serLc
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import joinRoot


logger = BasicLogging.getLogger(__name__)


class LcDataAdapter:
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def rowEquals(row, uid):
        pass

    @staticmethod
    @abstractmethod
    def initLcFrom(row):
        pass

    @staticmethod
    @abstractmethod
    def appendRow(times, mags, errors, row):
        pass


class K2Adapter(LcDataAdapter):
    """Parses Light curves from LSST csv K2 data. Ignore data having
    nonzero SAP_QUALITY.

    Col 0 - TIME [64-bit floating point] - The time at the mid-point of the
    cadence in BKJD. Kepler Barycentric Julian Day (BKJD) is Julian day minus
    2454833.0 (UTC=January 1, 2009 12:00:00) and corrected to be the arrival
    times at the barycenter of the Solar System.

    Col 7 - PDCSAP_FLUX [32-bit floating point] - The flux contained in the
    optimal aperture in electrons per second after the PDC module has applied
    its cotrending algorithm to the PA light curve. To better understand how
    PDC manipulated the light curve, read Section 2.3.1.2 and see the PDCSAPFL
    keyword in the header.

    Col 8 - PDCSAP_FLUX_ERR [32-bit floating point] - The 1-sigma error in PDC
    flux values.

    Col 9 - SAP_QUALITY [32-bit integer] - Flags containing information about
    the quality of the data. Table 2-3 explains the meaning of each active bit.
    See the Data Characteristics Handbook and Data Release Notes for more
    details on safe modes, coarse point, argabrightenings, attitude tweaks, etc.
    Unused bits are reserved for future use.

    # only select data where SAP quality flags are 0
    goodRows = np.where(data[:, 9] == 0)[0]
    """
    @staticmethod
    def rowEquals(row, uid):
        pass

    @staticmethod
    def initLcFrom(row):
        pass
        # times = data[goodRows, 0]
        # magnitudes = data[goodRows, 7]
        # errors = data[goodRows, 8]

    @staticmethod
    def appendRow(times, mags, errors, row):
        pass


class Ogle3Adapter(LcDataAdapter):
    @staticmethod
    def rowEquals(row, uid):
        return row[-1] == uid

    @staticmethod
    def initLcFrom(row):
        """Expects OGLE3 source data file to have the following columns:
         0=HJD, 1=MAG, 2=ERR, 3=FIELD, 4=LABEL, 5=NUM, 6=BAND, 7=ID"""
        return row[-1], row[4], [row[0]], [row[1]], [row[2]]

    @staticmethod
    def appendRow(times, mags, errors, row):
        """Expects OGLE3 source data file to have the following columns:
         0=HJD, 1=MAG, 2=ERR"""
        times.append(row[0])
        mags.append(row[1])
        errors.append(row[2])


class MachoAdapter(LcDataAdapter):
    """Macho column format:
    0 - macho_uid
    1 - classification
    2 - date_observed
    3 - magnitude
    4 - error
    """
    @staticmethod
    def rowEquals(row, uid):
        return row[0] == uid

    @staticmethod
    def initLcFrom(row):
        return row[0], row[1], [row[2]], [row[3]], [row[4]]

    @staticmethod
    def appendRow(times, mags, errors, row):
        times.append(row[2])
        mags.append(row[3])
        errors.append(row[4])


def loadFlatLcDataset(params, dbParams):
    """Loads and aggregates light curves from single csv file of individual data
    points storing results in a database."""
    dataPath = joinRoot(params["relativePath"])
    logger.info("Loading from: %s", dataPath)
    skiprows = params["skiprows"]
    dataLimit = params.get("dataLimit", float("inf"))
    table = dbParams["raw_lc_table"]
    commitFrequency = dbParams["commitFrequency"]

    dataName = params["dataName"]
    logger.info("Using %s LC adapter", dataName)
    if dataName == "ogle3":
        adapter = Ogle3Adapter
    elif dataName == "macho":
        adapter = MachoAdapter
    elif dataName == "k2":
        adapter = K2Adapter
    else:
        raise ValueError("Unsupported dataName: %s" % dataName)

    conn = connFromParams(dbParams)
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_LCS % table)
    reportTableCount(cursor, table, msg="before loading")
    insertOrReplaceQuery = INSERT_REPLACE_INTO_LCS % table
    with open(dataPath, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for _ in range(skiprows):
            next(reader)

        completedLcs = 0
        uid = label = times = mags = errors = None
        for i, row in enumerate(reader):
            if adapter.rowEquals(row, uid):
                # continue building current LC
                adapter.appendRow(times, mags, errors, row)
            else:
                if uid is not None:
                    # finish current LC, except for first time
                    args = (uid, label) + serLc(times, mags, errors)
                    cursor.execute(insertOrReplaceQuery, args)
                    completedLcs += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("completed lc with len: %s", len(times))

                    if not completedLcs % commitFrequency:
                        logger.info("committing progress: %s", completedLcs)
                        conn.commit()

                # initialize new LC
                uid, label, times, mags, errors = adapter.initLcFrom(row)

            if i >= dataLimit:
                break

    reportTableCount(cursor, table, msg="after loading")
    conn.commit()
    conn.close()
