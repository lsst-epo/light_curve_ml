import pickle

import numpy as np


def serLc(times, mags, errors) -> (bytes, bytes, bytes):
    """Serializes light curve attributes (as arrays or lists) to bytes objs
    """
    t = serArray(times)
    m = serArray(mags)
    e = serArray(errors)
    return t, m, e


def serArray(a: np.ndarray) -> bytes:
    return pickle.dumps(a)


def deserLc(times: bytes, mags: bytes, errors: bytes) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """Deserializes bytes light curves to numpy arrays of float64
    (equivalent of Python float)."""
    t = deserArray(times)
    m = deserArray(mags)
    e = deserArray(errors)
    return t, m, e


def deserArray(bytesObj: bytes) -> np.ndarray:
    bytesArray = pickle.loads(bytesObj, encoding="bytes")
    return np.array(bytesArray, dtype=np.float64)
