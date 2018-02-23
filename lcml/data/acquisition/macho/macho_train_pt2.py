import os
import re

import numpy as np

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import absoluteFilePaths


logger = BasicLogging.getLogger(__name__)


def main():
    outDir = os.path.join(os.environ["LSST"], "data/macho/class")
    allData = [",".join(["classification", "field_id", "tile_id", "sequence",
                         "date_observed", "red_magnitude", "red_error",
                         "blue_magnitude", "blue_error"]) + "\n"]
    pattern = r"""\d+"""
    for f in absoluteFilePaths(outDir, ext="csv"):
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


    trainFile = os.path.join(outDir, "macho-train.csv")
    with open(trainFile, "w") as f:
        f.writelines(allData)


if __name__ == "__main__":
    main()
