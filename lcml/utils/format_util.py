from datetime import datetime
from typing import Union

from astropy.time import Time

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def toDatetime(time: int, fmt: str="mjd", scale: str="tt") -> Union[datetime,
                                                                    None]:
    """Converts time in specified format and scale (e.g, Modified Julian Date
    (MJD) and Terrestrial Time) to datetime."""
    try:
        t = Time(float(time), format=fmt, scale=scale)
    except ValueError:
        logger.exception("Could not create time from: %s", time)
        return None

    return t.datetime


def fmtPct(a, b, places: int=2) -> str:
    """Given a ratio a / b formats a string representing the percentage."""
    _base = "{:.%s%%}" % places
    return _base.format(float(a) / float(b)) if b else "NaN"


def truncatedFloat(places: int) -> str:
    """Produces a string suitable for sprintf-style formatting of a float to
    specified number of places after decimal. E.g., 5 -> '%.5f' """
    return "%%.%sf" % places


if __name__ == "__main__":
    print(toDatetime(2015, fmt="jyear", scale="tcb"))
