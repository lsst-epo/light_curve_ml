#!/usr/bin/env python3
"""Main entry point for running lcml ML pipeline. Expects paths to pipeline conf
file and log file."""
import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import numpy as np

from lcml.utils.logging_manager import LoggingManager


def _pipelineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True,
                        help="relative path to pipeline conf")
    parser.add_argument("--logFileName", "-l", required=True,
                        help="name of log file in $LCML/logs")
    return parser.parse_args()


def main():
    args = _pipelineArgs()
    LoggingManager.initLogging(fileName=args.logFileName)
    logger = logging.getLogger(__name__)

    # N.B. importing here ensures no logger objects are created before the call
    # to `LoggingManager.initLogging`
    np.warnings.filterwarnings("ignore")
    from lcml.pipeline import fromRelativePath
    np.warnings.resetwarnings()
    pipe = fromRelativePath(args.path)
    try:
        pipe.runPipe()
    except BaseException:
        logger.exception("Unhandled exception from main pipeline, exiting.")


if __name__ == "__main__":
    main()
