import argparse
import multiprocessing
from multiprocessing import Pool
import time
import os

from feets.datasets.ogle3 import load_OGLE3_catalog, fetch_OGLE3
from feets.extractors.core import DATA_TIME
import requests

from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.error_handling import retry


logger = getBasicLogger(__name__, "peek_ogle.log")


SUFFICIENT_LENGTH = 60


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, default=0,
                        help="Begin index for sample of ogle3 ids to process")
    parser.add_argument("-e", "--end", type=int, default=int(10e6),
                        help="End index for sample of ogle3 ids to process")

    parser.add_argument("-l", "--sufficientLength", type=int,
                        default=SUFFICIENT_LENGTH, help="threshold on light "
                                                        "curve data length")
    return parser.parse_args()


@retry(timeoutSec=120, initialRetryDelaySec=1, maxRetryDelaySec=100,
       retryExceptions=(requests.RequestException,), logger=logger)
def fetchOgle3(vid):
    return fetch_OGLE3(vid)


def downOgle():
    sAll = time.time()
    args = _getArgs()
    df = load_OGLE3_catalog()
    ids = [i for i in df["ID"] if i != "-99.99"]

    validLc = 0
    tooShortLc = 0

    s = time.time()
    # feets library data structures fail in a multiprocess setting
    # pool = Pool(processes=multiprocessing.cpu_count())
    # datas = pool.map(ogleOrNone, ids[:args.limit])

    c = 0
    toProcess = args.end - args.start
    for vid in ids[args.start: args.end]:
        c += 1
        if c % 100 == 0:
            logger.info("%s / %s", c, toProcess)

        dataObj = fetchOgle3(vid)
        bunch = dataObj.data
        for bandChar, band in bunch.items():
            if len(band.time) > args.sufficientLength:
                validLc += 1
            else:
                tooShortLc += 1

    logger.info("\nScript total elapsed: %.2fs",time.time() - sAll)
    logger.info("Process data elapsed: %.2fs\n", time.time() - s)
    totalLcs = float(validLc + tooShortLc)
    logger.info("Processed %d / %s OGLE3 ids", toProcess, len(ids))
    logger.info("Found %d total light curves", totalLcs)
    validRate = "{:.2%}".format(validLc / totalLcs)
    insuffRate = "{:.2%}".format(tooShortLc / totalLcs)
    logger.info("Valid data: %s (%s)\nInsufficient length: %s (%s)", validLc,
        validRate, tooShortLc, insuffRate)


def reportDownloaded():
    _dir = [f for f in os.listdir("/Users/ryanjmccall/feets_data/ogle3")
            if f.endswith(".tar")]
    print("found %d tarfiles" % len(_dir))


if __name__ == "__main__":
    reportDownloaded()
    downOgle()
    reportDownloaded()
    # print(fetch_OGLE3("OGLE-LMC-ACEP-001").data.I.time)
