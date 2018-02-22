"""Obtains MACHO lightcurves using STILTS command-line tool."""
from collections import defaultdict
import logging
import os
import subprocess
from subprocess import CalledProcessError
import time

from prettytable import PrettyTable

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def genList(start, end):
    return list(range(start, end + 1))


def tapCommandBase(jreBinaryPath="/usr/bin/java"):
    jarPath = os.path.join(os.environ["LSST"], "jars/stilts.jar")
    commandBase = [jreBinaryPath, "-jar", jarPath, "tapquery"]
    return commandBase + ["tapurl=http://machotap.asvo.nci.org.au/ncitap/tap"]


def main():
    outDir = os.path.join(os.environ["LSST"], "data/macho/raw")
    commandBase = tapCommandBase()

    returnedLimit = 500000
    limit = int(10e7)
    # testQuery = "SELECT TOP 10 * FROM public.star_view"
    joinQuery = ("SELECT TOP %s b.poc, a.fieldid, a.tileid, a.seqn, " 
                 "a.obsid, a.dateobs, a.rmag, a.rerr, a.bmag, a.berr "
                 "FROM public.photometry_view AS a "
                 "JOIN public.varstar_view AS b "
                 "ON (a.fieldid=b.field AND a.tileid=b.tile AND a.seqn=b.seqn) "
                 "WHERE a.fieldid=%s AND b.poc='%s'")

    # Due to a limitation of returning at most 500K records at a time, the data
    # is grabbed across a series of queries for each observation field and for
    # each poc category
    # fields = [1, 2]

    # fields based on data shown at http://macho.nci.org.au/macho_photometry/
    fields = (genList(25, 180) + genList(206, 208) + genList(211, 213) +
              genList(301, 311) + genList(401, 403))
    categoryStart, categoryEnd = 1, 11
    classCounts = defaultdict(int)
    allStart = time.time()
    for field in fields:
        for cat in range(categoryStart, categoryEnd + 1):
            logger.info("Field: %s Class: %s", field, cat)
            outPath = os.path.join(outDir, "c%s_f%s.csv" % (cat, field))
            fullQuery = joinQuery % (limit, field, cat)
            cmd = commandBase + ["adql=" + fullQuery, "out=" + outPath,
                                 "compress=true"]
            apiStart = time.time()
            try:
                output = subprocess.check_output(cmd)
            except CalledProcessError:
                logger.exception("JAR call failed")
                return

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("call took: %.01fs", time.time() - apiStart)
                if output:
                    logger.debug("subprocess output: %s",
                                 output.decode("utf-8"))

            # if outfile is empty, print a warning and delete it
            with open(outPath, "r") as outFile:
                outLineCount = sum(1 for _ in outFile)

            classCounts[cat] += outLineCount
            if outLineCount == 1:
                logger.info("Skipping empty result")
                os.remove(outPath)

            if outLineCount >= returnedLimit:
                logger.warning("Reached TAP limit! Data likely missed: %s",
                               outLineCount)

    t = PrettyTable(["Category", "Counts", "Percentage"])
    totalCounts = sum(classCounts.values())
    for cat, counts in sorted(classCounts.items()):
        t.add_row([cat, counts, round(100.0 * counts / totalCounts, 2)])

    # +----------+---------+------------+
    # | Category | Counts | Percentage |
    # +----------+---------+------------+
    # | 1 | 2668376 | 32.29 |
    # | 2 | 612715 | 7.41 |
    # | 3 | 111089 | 1.34 |
    # | 4 | 619357 | 7.49 |
    # | 5 | 318188 | 3.85 |
    # | 6 | 55188 | 0.67 |
    # | 7 | 152359 | 1.84 |
    # | 8 | 352080 | 4.26 |
    # | 9 | 187325 | 2.27 |
    # | 10 | 3048492 | 36.89 |
    # | 11 | 138465 | 1.68 |
    # +----------+---------+------------+

    logger.info(t)
    logger.info("Entire harvest took: %.01fm", (time.time() - allStart) / 60)


if __name__ == "__main__":
    main()
