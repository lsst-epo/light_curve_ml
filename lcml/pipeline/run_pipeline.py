import argparse

from lcml.pipeline import fromRelativePath
from lcml.utils.basic_logging import BasicLogging


def _pipelineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="relative path to pipeline conf")
    parser.add_argument("--logFileName", required=True,
                        help="desired name of log file in $LSST/logs")
    return parser.parse_args()


def main():
    args = _pipelineArgs()
    BasicLogging.initLogging(fileName=args.logFileName)
    pipe = fromRelativePath(args.path)
    pipe.runPipe()


if __name__ == "__main__":
    main()
