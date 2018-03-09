from collections import Counter
import os
import re

import numpy as np

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import absoluteFilePaths


logger = BasicLogging.getLogger(__name__)


def main():
    # FIXME Create this format instead:
    """Macho column format:
    0 - macho_uid
    1 - classification
    2 - date_observed
    3 - magnitude
    4 - error
    """

    inDir = os.path.join(os.environ["LSST"], "data/macho/class")
    allData = [",".join(["classification", "field_id", "tile_id", "sequence",
                         "date_observed", "red_magnitude", "red_error",
                         "blue_magnitude", "blue_error"]) + "\n"]
    pattern = r"""\d+"""
    dataLengths = Counter()
    missing = [",".join(("field", "tile", "seqn")) + "\n"]
    for f in absoluteFilePaths(inDir, ext="csv"):
        try:
            data = np.loadtxt(f, skiprows=1, delimiter=",")
        except ValueError:
            logger.critical("can't load file: %s", f)
            continue

        fileName = f.split("/")[-1].split(".")[0]
        field, tile, seqn, classif = re.findall(pattern, fileName)
        prefix = [classif, field, tile, seqn]
        for row in data:
            allData.append(",".join(prefix + [str(x) for x in row]) + "\n")

        dataLengths[len(data) // 10] += 1
        if not len(data):
            missing.append(",".join((field, tile, seqn)) + "\n")

    outDir = os.path.join(os.environ["LSST"], "data/macho")
    trainFile = os.path.join(outDir, "macho-train.csv")
    with open(trainFile, "w") as f:
        f.writelines(allData)

    missingFile = os.path.join(outDir, "train-fails.csv")
    with open(missingFile, "w") as f:
        f.writelines(missing)

    logger.critical("LC length distribution: %s",
                    sorted(list(dataLengths.items())))


if __name__ == "__main__":
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'Empty input file')
        main()
