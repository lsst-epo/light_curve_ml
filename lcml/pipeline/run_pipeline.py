#!/usr/bin/env python3
"""Main entry point for running lcml ML pipeline. Expects paths to pipeline conf
file and log file."""
import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np

np.warnings.filterwarnings("ignore")
from lcml.pipeline import fromRelativePath
np.warnings.resetwarnings()

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def _pipelineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True,
                        help="relative path to pipeline conf")
    parser.add_argument("--logFileName", "-l", required=True,
                        help="name of log file in $LCML/logs")
    return parser.parse_args()


def main():
    args = _pipelineArgs()
    BasicLogging.initLogging(fileName=args.logFileName)
    pipe = fromRelativePath(args.path)
    try:
        pipe.runPipe()
    except BaseException:
        logger.exception("Unhandled exception from main pipeline, exiting.")


if __name__ == "__main__":
    main()
