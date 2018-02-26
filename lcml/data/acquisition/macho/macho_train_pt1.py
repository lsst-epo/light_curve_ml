"""Obtains MACHO lightcurves using STILTS command-line tool."""
from collections import defaultdict
import os
import subprocess
from subprocess import CalledProcessError

import numpy as np
from prettytable import PrettyTable

from lcml.data.acquisition.macho.macho_from_stilts import tapCommandBase
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def genList(start, end):
    return list(range(start, end + 1))


def main():
    inPath = os.path.join(os.environ["LSST"],
                          "data/macho/macho-classifications.csv")

    outDir = os.path.join(os.environ["LSST"], "data/macho/class")

    commandBase = tapCommandBase()
    query = ("SELECT dateobs, rmag, rerr, bmag, berr "
             "FROM public.photometry_view "
             "WHERE fieldid=%s AND tileid=%s AND seqn=%s")

    classCounts = defaultdict(int)
    classData = np.loadtxt(fname=inPath, dtype=int, delimiter=",",
                           skiprows=1)
    logger.critical("processing %d requests", len(classData))
    smallestLcNoRetry = 20
    for field, tile, seqn, classif in classData:
        classCounts[classif] += 1
        fname = "field=%s_tile=%s_seqn=%s_class=%s" % (field, tile, seqn,
                                                       classif)
        outPath = os.path.join(outDir, fname + ".csv")
        if os.path.exists(outPath):
            _tempData = np.loadtxt(outPath, dtype=str, delimiter=",")
            if len(_tempData) > smallestLcNoRetry:
                # skip downlaod if we already have a file with sufficient data
                # logger.critical("skipping %s", fname)
                continue

        logger.critical(outPath)
        fullQuery = query % (field, tile, seqn)
        cmd = commandBase + ["adql=" + fullQuery, "out=" + outPath]
        try:
            subprocess.check_output(cmd)
        except CalledProcessError:
            logger.exception("JAR call failed")
            continue

    # +----------+--------+------------+
    # | Category | Counts | Percentage |
    # +----------+--------+------------+
    # | 1 | 7405 | 34.48 |
    # | 2 | 1765 | 8.22 |
    # | 3 | 315 | 1.47 |
    # | 4 | 1185 | 5.52 |
    # | 5 | 683 | 3.18 |
    # | 6 | 315 | 1.47 |
    # | 7 | 822 | 3.83 |
    # | 8 | 1134 | 5.28 |
    # | 9 | 778 | 3.62 |
    # | 10 | 6835 | 31.83 |
    # | 11 | 237 | 1.1 |
    # +----------+--------+------------+
    t = PrettyTable(["Category", "Counts", "Percentage"])
    totalCounts = sum(classCounts.values())
    for cat, counts in sorted(classCounts.items()):
        t.add_row([cat, counts, round(100.0 * counts / totalCounts, 2)])
    logger.critical("\n" + str(t))


if __name__ == "__main__":
    main()
