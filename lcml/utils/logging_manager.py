from datetime import datetime
import logging
import sys
from typing import Union

from pytz import timezone, utc

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


def nameToLevel(name: Union[str, int]) -> int:
    """Converts the English name for logging level to the logging module's
    internal integer code"""
    return _levelNames[name.upper()] if isinstance(name, str) else name


def levelToName(level: int) -> str:
    return _levelNames[level]


def getLogFormat(name: str) -> str:
    if name == "message":
        return MESSAGE_FORMAT
    elif name == "time-message":
        return TIME_MESSAGE_FORMAT
    else:
        return DEFAULT_FORMAT


class LoggingManager:
    """Manages app logging configuration. Reads config file with following
    keys:
    1) 'basicConfig' - values passed to `logging.basicConfig`
    2) 'handlers' - handler definitions with 'type' attributes either 'stream'
    or 'file'
    3) 'modules' - list of module specific logger level settings
    See `conf/common/logging.json` for an example
    """
    # configuration used by manager, mainly for debugging
    _config = None

    @classmethod
    def initLogging(cls, fileName: str=None, fmt: str=None,
                    config: dict=None):
        """Initializes logging across app. Must be called before logger objects
        are created. Configuration read from
        `$LCML/conf/common/logging.json`.

        :param fileName: name of log file written to by FileHandler
        :param fmt: logger format override
        :param config: replacement for default logging config
        """
        cls._config = config if config else jsonConfig("logging.json")
        if cls._config.get("active", True):
            # Python libraries may specify NullHandlers; however, this adds them
            # to the root logger. Its having 1 or more handlers effectively
            # prevents `logging.basicConfig()` from doing anything!
            # So to activate logging, these handlers must first be cleared
            logging.root.handlers = []

        # time-zone conversion
        tz = timezone(cls._config["tz"])
        def localConverter(*args):
            return utc.localize(datetime.utcnow()).astimezone(tz).timetuple()
        logging.Formatter.converter = localConverter

        # set up kwargs for basicConfig()
        kwargs = cls._config["basicConfig"]
        kwargs["level"] = nameToLevel(kwargs["level"])

        _format = getLogFormat(fmt if fmt else kwargs["format"])
        kwargs["format"] = _format

        # handlers
        handlers = []
        for defn in cls._config["handlers"]:
            hdlrType = defn["type"].lower()
            if hdlrType == "stream":
                hdlr = logging.StreamHandler(sys.stdout)
                hdlr.setLevel(nameToLevel(defn["level"]))
                hdlr.setFormatter(logging.Formatter(_format))
            elif hdlrType == "file":
                fileName = fileName if fileName else defn["filename"]
                fullFileName = joinRoot("logs", fileName)
                hdlr = logging.FileHandler(filename=fullFileName,
                                           mode=defn.get("mode", "a"))
            else:
                raise ValueError("bad handler type: " + hdlrType)
            handlers.append(hdlr)

        kwargs["handlers"] = handlers

        logging.basicConfig(**kwargs)

        # module-specific logger levels
        for setting in cls._config["modules"]:
            level = nameToLevel(setting["level"])
            logging.getLogger(setting["module"]).setLevel(level)
