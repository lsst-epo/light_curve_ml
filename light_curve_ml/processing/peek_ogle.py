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
    insufficient = 0
    bunchNone = 0
    pool = Pool(processes=multiprocessing.cpu_count())
    s = time.time()
    bunches = pool.map(fetchWork, ids[:args.limit])
    for bunch in bunches:
        if bunch and bunch.bands:
            for subBunch in bunch.bands.values():
                if len(subBunch[DATA_TIME]) > SUFFICIENT_LENGTH:
                    validData += 1
                else:
                    insufficient += 1

        elif bunch is None:
            bunchNone += 1

    print("Computation elapsed: %.2fs" % (time.time() - s))
    print("All elapsed: %.2fs" % (time.time() - sAll))
    print("Total OGLE3 ids: %s" % len(ids))
    validRate = "{:.3%}".format(validData / len(ids))
    insuffRate = "{:.3%}".format(insufficient / len(ids))
    print("Valid: %s (%s) insufficient length: %s (%s) bunchNone: %s" % (
        validData, validRate, insufficient, insuffRate, bunchNone))


if __name__ == "__main__":
    peekOgle()