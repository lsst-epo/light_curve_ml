"""Obtains MACHO lightcurves using STILTS command-line tool."""
from collections import defaultdict
import os
import subprocess
from subprocess import CalledProcessError
import time


def genList(start, end):
    return list(range(start, end + 1))


def main():
    jreBinary = "/usr/bin/java"
    jarPath = os.path.join(os.environ["LSST"], "jars/stilts.jar")
    outDir = os.path.join(os.environ["LSST"], "data/macho/raw")
    commandBase = [jreBinary, "-jar", jarPath, "tapquery"]
    commandBase += ["tapurl=http://machotap.asvo.nci.org.au/ncitap/tap"]

    queryLimit = 500000
    limit = int(10e7)
    # testQuery = "SELECT TOP 10 * FROM public.star_view"
    joinQuery = ("SELECT TOP %s b.classification, a.fieldid, a.tileid, a.seqn, " 
                 "a.obsid, a.dateobs, a.rmag, a.rerr, a.bmag, a.berr "
                 "FROM public.photometry_view AS a "
                 "JOIN public.varstar_view AS b "
                 "ON (a.fieldid=b.field AND a.tileid=b.tile AND a.seqn=b.seqn) "
                 "WHERE a.fieldid=%s AND b.classification='%s'")

    # Due to a limitation of returning at most 500K records at a time, the data
    # is grabbed across a series of queries for each observation field and for
    # each classification category
    # fields = [1, 2]

    # fields based on data shown at http://macho.nci.org.au/macho_photometry/
    fields = (genList(1, 180) + genList(206, 208) + genList(211, 213) +
              genList(301, 311) + genList(401, 403))
    categoryStart, categoryEnd = 1, 11
    classCounts = defaultdict(int)
    allStart = time.time()
    for field in fields:
        for cat in range(categoryStart, categoryEnd + 1):
            print("\nField: %s Class: %s" % (field, cat))
            outPath = os.path.join(outDir, "c%s_f%s.csv" % (cat, field))
            fullQuery = joinQuery % (limit, field, cat)
            cmd = commandBase + ["adql=" + fullQuery, "out=" + outPath,
                                 "compress=true"]
            apiStart = time.time()
            try:
                output = subprocess.check_output(cmd)
            except CalledProcessError as e:
                print(e)
                return

            print("call took: %.01fs" % (time.time() - apiStart))
            if output:
                print("subprocess output: %s" % output.decode("utf-8"))

            # if outfile is empty, print a warning and delete it
            with open(outPath, "r") as outFile:
                outLineCount = sum(1 for _ in outFile)

            classCounts[cat] += outLineCount
            if outLineCount == 1:
                print("result: %s too small" % field)
                os.remove(outPath)

            if outLineCount >= queryLimit:
                print("WARNING: Reach limit! Data likely missed: %s" %
                      outLineCount)

    print("cat\tcounts\tpercentage")
    totalCounts = sum(classCounts.values())
    for cat, counts in sorted(classCounts.items()):
        print("%s\t%s\t%.02f" % (cat, counts, 100 * counts / totalCounts))

    print("Entire harvest took: %.01fm" % (time.time() - allStart) / 60)


if __name__ == "__main__":
    main()
