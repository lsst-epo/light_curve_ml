import argparse
import multiprocessing
from multiprocessing import Pool
import time

from feets.datasets import macho
from feets.datasets.ogle3 import load_OGLE3_catalog, fetch_OGLE3
from feets.extractors.core import DATA_TIME
import requests


def peekMacho():
    """Only provides small number of macho ids"""
    print(macho.available_MACHO_lc())


def fetchWork(vid):
    try:
        return fetch_OGLE3(vid)
    except requests.RequestException:
        return None


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--limit", type=int, default=int(1e20),
                        help="Limit script to running first n things")
    return parser.parse_args()


def peekOgle():
    sAll = time.time()
    args = _getArgs()
    df = load_OGLE3_catalog()
    ids = [id for id in df["ID"] if id != "-99.99"]

    SUFFICIENT_LENGTH = 50
    validData = 0
    insufficientData = 0
    bunchNoneData = 0
    pool = Pool(processes=multiprocessing.cpu_count())
    s = time.time()
    bunches = pool.map(fetchWork, ids[:args.limit])
    for bunch in bunches:
        if bunch and bunch.bands:
            for subBunch in bunch.bands.values():
                if len(subBunch[DATA_TIME]) > SUFFICIENT_LENGTH:
                    validData += 1
                else:
                    insufficientData += 1

        elif bunch is None:
            bunchNoneData += 1

    print("\nScript total elapsed: %.2fs" % (time.time() - sAll))
    print("Process data elapsed: %.2fs\n" % (time.time() - s))
    totalProcessed = float(validData + insufficientData + bunchNoneData)
    print("Processed %d of %s total OGLE3 ids" % (totalProcessed, len(ids)))
    validRate = "{:.2%}".format(validData / totalProcessed)
    insuffRate = "{:.2%}".format(insufficientData / totalProcessed)
    print("Valid data: %s (%s)\nInsufficient length: %s (%s)\nBunch None: %s" %
          (validData, validRate, insufficientData, insuffRate, bunchNoneData))



if __name__ == "__main__":
    peekOgle()