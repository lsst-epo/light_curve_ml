import logging
import os
import sys


LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


DATE_FORMAT = "%Y.%m.%d %H:%M:%S"


def getBasicLogger(name, fileName,
                   format=LOG_FORMAT,
                   dateFmt=DATE_FORMAT,
                   level=logging.INFO):
    outFile = fileName.split(os.path.sep)[-1]
    outFile = outFile.replace("py", "log")
    logPath = os.path.join(os.environ.get("LSST"), "logs", outFile)
    logging.basicConfig(filename=logPath,
                        filemode="a",
                        format=format,
                        datefmt=dateFmt,
                        level=level)

    logger = logging.getLogger(name)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)

    return logger
