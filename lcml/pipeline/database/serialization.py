import pickle

import numpy as np


def serLc(times, mags, errors):
    """Serializes light curve attributes (as arrays or lists) to binary strings
    """
    t = serArray(times)
    m = serArray(mags)
    e = serArray(errors)
    return t, m, e


def serArray(a):
    return pickle.dumps(a)


def deserLc(times, mags, errors):
    """Deserializes 3 binary-string light curves to 3 numpy arrays of float64
    (equivalent of Python float)."""
    t = deserArray(times)
    m = deserArray(mags)
    e = deserArray(errors)
    return t, m, e


def deserArray(binStr):
    return np.array(pickle.loads(binStr.encode("utf-8"), encoding="bytes"),
                    dtype=np.float64)
