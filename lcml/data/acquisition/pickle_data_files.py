#!/usr/bin/env python3
import pickle

import numpy as np

from lcml.utils.context_util import joinRoot, loadJson


def main():
    paths = ["data/macho/macho-sample.csv",
             "data/ucr_lcs/StarLightCurves_TEST.csv"]
    for path in paths:
        fullPath = joinRoot(path)
        ext  = path.split(".")[-1]
        if ext == "csv":
            obj = loadCsv(fullPath)
        elif ext == "json":
            obj = loadJson(fullPath)
        else:
            print("bad ext: " + ext)
            continue

        dumpWhereFound(obj, fullPath, ext)


def loadCsv(p):
   return np.genfromtxt(open(p, "rb"), delimiter=",", dtype=None, encoding=None)


def dumpWhereFound(obj, sourcePath: str, extension: str):
    """dumps pickled version of object in the same place as its source path

    :param obj - python object
    :param sourcePath - full path of the file from which the object was created
    :param extension - original file's extension
    """
    dumpPath = sourcePath.replace(extension, "pkl")
    print("dumping: " + dumpPath)
    with open(dumpPath, "wb") as outFile:
        pickle.dump(obj, outFile, protocol=3)


if __name__ == "__main__":
    main()
