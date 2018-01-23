import argparse
import multiprocessing
from multiprocessing import Pool
import tarfile
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
    try:
        return fetch_OGLE3(vid)
    except tarfile.ReadError:
        print("error reading tarfile: %s" % vid)
        raise


_REPORT_FREQ = 100


def downOgle():
    sAll = time.time()
    args = _getArgs()
    df = load_OGLE3_catalog()
    ids = [i for i in df["ID"] if i != "-99.99"]
    print("Found %d OGLE3 ids" % len(ids))

    validLc = 0
    tooShortLc = 0

    s = time.time()
    # feets library data structures fail in a multiprocess setting
    # pool = Pool(processes=multiprocessing.cpu_count())
    # datas = pool.map(ogleOrNone, ids[:args.limit])

    c = args.start
    for vid in ids[args.start: args.end]:
        if c % _REPORT_FREQ == 0:
            logger.info("%s / %s", c, args.end)

        c += 1
        dataObj = fetchOgle3(vid)
        for bandChar, band in dataObj.data.items():
            if len(band.time) > args.sufficientLength:
                validLc += 1
            else:
                tooShortLc += 1

    logger.info("\nScript total elapsed: %.2fs",time.time() - sAll)
    logger.info("Process data elapsed: %.2fs\n", time.time() - s)
    totalLcs = float(validLc + tooShortLc)
    processed = args.end - args.start
    logger.info("Processed %d / %s OGLE3 ids", processed, len(ids))
    logger.info("Found %d total light curves", totalLcs)
    validRate = "{:.2%}".format(validLc / totalLcs)
    insuffRate = "{:.2%}".format(tooShortLc / totalLcs)
    logger.info("Valid data: %s (%s)\nInsufficient length: %s (%s)", validLc,
        validRate, tooShortLc, insuffRate)


def reportDownloaded():
    downDir = os.path.join(os.path.expandvars("$HOME"), "feets_data/ogle3")
    _dir = [f for f in os.listdir(downDir)
            if f.endswith(".tar")]
    print("found %d tarfiles" % len(_dir))


if __name__ == "__main__":
    reportDownloaded()
    downOgle()
    reportDownloaded()
    # print(fetch_OGLE3("OGLE-LMC-ACEP-001").data.I.time)
