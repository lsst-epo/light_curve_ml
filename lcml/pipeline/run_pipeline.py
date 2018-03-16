import argparse

from lcml.pipeline import fromRelativePath
from lcml.utils.basic_logging import BasicLogging


BasicLogging.initLogging()
logger = BasicLogging.getLogger(__name__)


def _pipelineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="relative path to pipeline conf")
    return parser.parse_args()


def main():
    args = _pipelineArgs()
    pipe = fromRelativePath(args.path)
    pipe.runPipe()


if __name__ == "__main__":
    main()
