from collections import Counter
import os
import re

import numpy as np

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import absoluteFilePaths, joinRoot


logger = BasicLogging.getLogger(__name__)


#: Source: MACHO Tap server (http://machotap.asvo.nci.org.au/ncitap/tap),
#: table 'public.varstar_view', see column 'classification' description
MACHO_NUM_TO_LABEL = {'1': "rrlyrae-ab", '2': "rrlyrae-c", '3': "rrlyrae-e",
                      '4': "ceph-fundam", '5': "ceph-1st-overtone",
                      '6': "lpv-wood-a", '7': "lpv-wood-b", '8': "lpv-wood-c",
                      '9': "lpv-wood-d", '10': "eclips-binary",
                      '11': "rrlyrae-plus-gb"}


def main():
    """Generates a .csv file containing the labeled MACHO training set.
    Columns of macho-train.csv output:
    0 - macho_uid
    1 - classification
    2 - date_observed
    3 - magnitude
    4 - error

    Additionally generates a second csv file containing the UIDs of missing data
    files.
    """
    inDir = joinRoot("data/macho/class")
    redBands = [",".join(["field-tile-seqn-band", "classLabel", "date_observed",
                          "magnitude", "error"]) + "\n"]
    blueBands = []

    # N.B. pt1 generated file names of the form:
    # 'field=1_tile=33_seqn=10_class=6.csv'
    pattern = r"""\d+"""
    dataLengths = Counter()

    # Heading for missing UID file
    missing = [",".join(("field", "tile", "seqn")) + "\n"]
    for f in absoluteFilePaths(inDir, ext="csv"):
        try:
            data = np.loadtxt(f, skiprows=1, delimiter=",")
        except ValueError:
            logger.critical("can't load file: %s", f)
            continue

        fileName = f.split("/")[-1].split(".")[0]
        field, tile, seqn, classNum = re.findall(pattern, fileName)
        label = MACHO_NUM_TO_LABEL[classNum]
        prefix = [field, tile, seqn]
        for r in data:
            # column format for source file
            # 0=dateobs, 1=rmag, 2=rerr, 3=bmag, 4=berr

            # uid, class label, dateobs, rmag, rerr
            _rVals = [machoUid(prefix + ["R"]), label] + [str(_) for _ in r[:3]]

            # uid, class label, dateobs, bmag, berr
            _bVals = ([machoUid(prefix + ["B"]), label] + [str(r[0])] +
                      [str(_) for _ in r[3:]])
            redBands.append(",".join(_rVals) + "\n")
            blueBands.append(",".join(_bVals) + "\n")

        dataLengths[len(data) // 10] += 1  # data length histogram in 10s
        if not len(data):
            missing.append(",".join((field, tile, seqn)) + "\n")

    outDir = joinRoot("data/macho")
    trainFile = os.path.join(outDir, "macho-train.csv")
    with open(trainFile, "w") as f:
        f.writelines(redBands)
        f.writelines(blueBands)

    missingFile = os.path.join(outDir, "macho-train-fails.csv")
    with open(missingFile, "w") as f:
        f.writelines(missing)

    logger.critical("LC length distribution: %s",
                    sorted(list(dataLengths.items())))


def machoUid(*args):
    return "-".join(*args)


if __name__ == "__main__":
    main()
