"""Run a particular light curve against feets features detecting failures"""
import argparse
from datetime import timedelta
import time
from typing import Generator, List

from feets import FeatureSpace
from feets.extractors import registered_extractors
import numpy as np

from lcml.pipeline.database import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.database.serialization import deserLc
from lcml.pipeline.database.sqlite_db import connFromParams


DB_TIMEOUT = 60
TABLE_NAME = "clean_lcs"


def _clargs():
    p = argparse.ArgumentParser()
    p.add_argument("--dbPath", "-p", required=True,
                   help="rel path to sqlite db")
    p.add_argument("--id", "-i", required=True, help="LC ID")
    p.add_argument("--feature", "-f", help="Test LC against specific feature")
    return p.parse_args()


def featuresByData(data: List[str]) -> Generator[str, None, None]:
    """Return feets features supported by specified data types"""
    for fname, f in registered_extractors().items():
        if not f.get_data().difference(data):
            yield fname


def main():
    start = time.time()
    args = _clargs()

    dbParams = {"dbPath": args.dbPath, "timeout": DB_TIMEOUT}
    conn = connFromParams(dbParams)
    cursor = conn.cursor()

    _SELECT_SQL = "SELECT * FROM %s WHERE id=\"%s\"" % (TABLE_NAME, args.id)
    cursor.execute(_SELECT_SQL)
    try:
        row = next(cursor)
    except StopIteration:
        print("Found no LCs!")
        return

    times, mag, err = deserLc(*row[2:])
    conn.close()

    i = 0
    skipped = list()
    # fts = registered_extractors()
    fts = featuresByData(STANDARD_INPUT_DATA_TYPES)
    if args.feature:
        # Option to test specific feature only
        assert args.feature in fts
        fts = [args.feature]

    for i, featureName in enumerate(fts):
        fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, only=[featureName])
        try:
            fts, values = fs.extract(times, mag, err)
        except BaseException as e:
            print(e)
            print("failed for feature: %s")
            break

        if len(fts) and len(values):
            msg = "OK" if np.isfinite(values[0]) else "NOT FINITE!"
            print("%s %s: %s" % (msg, fts[0], values[0]))
        else:
            skipped.append(featureName)

    print("total %s skipped: %s" % (i, len(skipped)))
    print("skipped: %s" % skipped)
    print("elapsed: %s" % timedelta(seconds=time.time()-start))

if __name__ == "__main__":
    main()
