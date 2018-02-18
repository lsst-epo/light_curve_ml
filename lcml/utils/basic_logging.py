from datetime import datetime
import logging
import os
from pytz import timezone, utc
import sys

from lcml.utils.context_util import joinRoot, jsonConfig


DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
TIME_MESSAGE_FORMAT = "%(asctime)s - %(message)s"
MESSAGE_FORMAT = "%(message)s"
DATE_FORMAT = "%Y.%m.%d %H:%M:%S"

DEFAULT_BACKUPS = 20
DEFAULT_MAX_BYTES = 50e7 # 50MB


_levelNames = {
    logging.CRITICAL : 'CRITICAL',
    logging.ERROR : 'ERROR',
    logging.WARNING : 'WARNING',
    logging.INFO : 'INFO',
    logging.DEBUG : 'DEBUG',
    logging.NOTSET : 'NOTSET',
    'CRITICAL' : logging.CRITICAL,
    'ERROR' : logging.ERROR,
    'WARN' : logging.WARNING,
    'WARNING' : logging.WARNING,
    'INFO' : logging.INFO,
    'DEBUG' : logging.DEBUG,
    'NOTSET' : logging.NOTSET,
}


def nameToLevel(name):
    """Converts the English name for logging level to the logging module's
    internal integer code"""
    return _levelNames[name.upper()]


def levelToName(level):
    return _levelNames[level]


class BasicLogging:
    _config = None
    _consoleLevel = None
    _format = None

    @classmethod
    def initLogging(cls, fmt=None):
        cls._config = jsonConfig("logging.json")  # store conf for debugging
        basicParams = cls._config["basicParams"]  # kwargs for basicConfig()

        # logging.json control of main logging attributes
        basicParams["level"] = nameToLevel(basicParams["level"])
        globalFormat = fmt if fmt else basicParams["format"].lower()
        if globalFormat == "message":
            basicParams["format"] = MESSAGE_FORMAT
        elif globalFormat == "time-message":
            basicParams["format"] = TIME_MESSAGE_FORMAT
        else:
            basicParams["format"] = DEFAULT_FORMAT

        cls._format = basicParams["format"]
        basicParams["filename"] = joinRoot("logs", basicParams["filename"])
        if cls._config["active"]:
            # Python libraries may specify NullHandlers; however, this adds them
            # to the root logger. Its having 1 or more handlers effectively
            # prevents `logging.basicConfig()` from doing anything!
            # So to activate logging, these handlers must first be cleared
            logging.root.handlers = []

        logging.basicConfig(**basicParams)

        tz = timezone(cls._config["tz"])
        def localConverter(*args):
            return utc.localize(datetime.utcnow()).astimezone(tz).timetuple()

        cls._consoleLevel = nameToLevel(cls._config["consoleLevel"])
        logging.Formatter.converter = localConverter
        for setting in cls._config["modules"]:
            level = nameToLevel(setting["level"])
            logging.getLogger(setting["module"]).setLevel(level)

    @classmethod
    def getLogger(cls, name):
        logger = logging.getLogger(name)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(cls._consoleLevel)
        handler.setFormatter(logging.Formatter(cls._format))
        logger.addHandler(handler)
        return logger


def getBasicLogger(name, fileName,
                   logFormat=DEFAULT_FORMAT,
                   dateFmt=DATE_FORMAT,
                   level=logging.INFO):
    outFile = fileName.split(os.path.sep)[-1]
    outFile = outFile.replace("py", "log")
    logPath = os.path.join(os.environ.get("LSST"), "logs", outFile)
    logging.basicConfig(filename=logPath,
                        filemode="a",
                        format=logFormat,
                        datefmt=dateFmt,
                        level=level)

    logger = logging.getLogger(name)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(logFormat))
    logger.addHandler(handler)

    return logger
