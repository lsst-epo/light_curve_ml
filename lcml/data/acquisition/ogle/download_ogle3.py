#!/usr/bin/env python3
import argparse
import logging
import os
import tarfile
import time

from feets.datasets.ogle3 import load_OGLE3_catalog, fetch_OGLE3
import requests

from lcml.utils.error_handling import retry


logger = logging.getLogger(__name__)


SUFFICIENT_LENGTH = 60


_REPORT_FREQ = 100


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


def reportDownloaded():
    downDir = os.path.join(os.path.expandvars("$HOME"), "feets_data/ogle3")
    _dir = [f for f in os.listdir(downDir)
            if f.endswith(".tar")]
    print("found %d tarfiles" % len(_dir))


@retry(timeoutSec=120, initialRetryDelaySec=1, maxRetryDelaySec=100,
       retryExceptions=(requests.RequestException,))
def fetchOgle3(vid):
    result = None
    try:
        result = fetch_OGLE3(vid)
    except tarfile.ReadError:
        print("error reading tarfile: %s" % vid)
    except ValueError as e:
        print("Error occurred fetching file: %s" % e)
        print("Skipping vid: %s" % vid)

    return result


def main():
    reportDownloaded()
    sAll = time.time()
    args = _getArgs()
    df = load_OGLE3_catalog()
    ids = [i for i in df["ID"] if i != "-99.99"]
    print("Found %d OGLE3 ids" % len(ids))
    requestedIds = ids[args.start: args.end]

    validLc = 0
    tooShortLc = 0

    s = time.time()
    # pool = Pool(processes=multiprocessing.cpu_count())
    # datas = pool.map(fetchOgle3, requestedIds)

    c = args.start
    for vid in requestedIds:
        if c % _REPORT_FREQ == 0:
            logger.info("%s / %s", c, args.end)

        c += 1
        dataObj = fetchOgle3(vid)
        if dataObj:
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

    reportDownloaded()


if __name__ == "__main__":
    main()
