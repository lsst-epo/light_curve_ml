import argparse

import numpy as np

np.warnings.filterwarnings("ignore")
from lcml.pipeline import fromRelativePath
np.warnings.resetwarnings()

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def _pipelineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="relative path to pipeline conf")
    parser.add_argument("--logFileName", required=True,
                        help="desired name of log file in $LCML/logs")
    return parser.parse_args()


def main():
    args = _pipelineArgs()
    BasicLogging.initLogging(fileName=args.logFileName)
    pipe = fromRelativePath(args.path)
    try:
        pipe.runPipe()
    except BaseException:
        logger.exception("Exiting due to unhandled exception from main pipeline"
                         )


if __name__ == "__main__":
    main()
