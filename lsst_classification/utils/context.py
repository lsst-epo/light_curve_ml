import os


_ROOT_DIR = os.environ.get("LSST", None)
assert _ROOT_DIR is not None, "Please set the 'LSST' environment variable."
assert os.path.isdir(_ROOT_DIR), "Root dir: %s does not exist" % _ROOT_DIR


def rootDir():
    return _ROOT_DIR


def joinRoot(*paths):
    """Given path that is relative to the root dir, returns the absolute path"""
    return os.path.join(_ROOT_DIR, *paths)
