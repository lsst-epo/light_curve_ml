"""Obtains MACHO lightcurves using STILTS command-line tool."""
import os
import subprocess
from subprocess import CalledProcessError
import time


def main():
    # TODO only returning 500K rows

    jreBinary = "/usr/bin/java"
    jarPath = os.path.join(os.environ["LSST"], "jars/stilts.jar")
    outDir = os.path.join(os.environ["LSST"], "data/macho/raw")
    commandBase = [jreBinary, "-jar", jarPath, "tapquery"]
    commandBase += ["tapurl=http://machotap.asvo.nci.org.au/ncitap/tap"]

    subprocessTimeout = 60
    apiStartTime = time.time()
    testQuery = "SELECT TOP 1000 * FROM public.star_view"

    joinQuery = ("SELECT b.classification, a.fieldid, a.tileid, a.seqn, " 
                 "a.obsid, a.dateobs, a.rmag, a.rerr, a.bmag, a.berr "
                 "FROM public.photometry_view AS a "
                 "JOIN public.varstar_view AS b "
                 "ON (a.fieldid=b.field AND a.tileid=b.tile AND a.seqn=b.seqn) "
                 "WHERE a.fieldid=%s")

    fieldLimit = 180  # 403
    for i in range(1, fieldLimit):
        outPath = os.path.join(outDir, "f%s.csv" % i)
        cmd = commandBase + ["adql=" + joinQuery % i,
                             "out=" + outPath,
                             "compress=true"]
        try:
            output = subprocess.check_output(cmd)
        except CalledProcessError as e:
            print(e)
            return

        decodedOutput = output.decode("utf-8")
        print(decodedOutput)


if __name__ == "__main__":
    main()
