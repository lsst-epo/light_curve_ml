from astropy.time import Time


def toDatetime(time, format="mjd", scale="tt"):
    """Converts time in specified format and scale (e.g, Modified Julian Date
    (MJD) and Terrestrial Time) to datetime."""
    try:
        t = Time(float(time), format=format, scale=scale)
    except ValueError:
        print("Could not create time from: %s" % time)
        return None

    return t.datetime


def fmtPct(a, b, places=2):
    """Given a ratio a / b formats a string representing the percentage."""
    _base = "{:.%s%%}" % places
    return _base.format(float(a) / float(b)) if b else "NaN"


def truncatedFloat(places):
    """Produces a string suitable for sprintf-style formatting of a float to
    specified number of places after decimal. E.g., 5 -> '%.5f' """
    return "%%.%sf" % places


if __name__ == "__main__":
    print(toDatetime(2015, format="jyear", scale="tcb"))
