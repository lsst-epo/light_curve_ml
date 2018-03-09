from collections import Counter
import os
import re

import numpy as np

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import absoluteFilePaths


logger = BasicLogging.getLogger(__name__)


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
    inDir = os.path.join(os.environ["LSST"], "data/macho/class")
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
        field, tile, seqn, label = re.findall(pattern, fileName)
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

    outDir = os.path.join(os.environ["LSST"], "data/macho")
    trainFile = os.path.join(outDir, "macho-train.csv")
    with open(trainFile, "w") as f:
        f.writelines(redBands)
        f.writelines(blueBands)

    missingFile = os.path.join(outDir, "train-fails.csv")
    with open(missingFile, "w") as f:
        f.writelines(missing)

    logger.critical("LC length distribution: %s",
                    sorted(list(dataLengths.items())))


def machoUid(*args):
    return "-".join(*args)


if __name__ == "__main__":
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'Empty input file')
        main()
