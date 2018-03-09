import json
import os


_ROOT_DIR = os.environ.get("LSST", None)
assert _ROOT_DIR is not None, "Please set the 'LSST' environment variable."
assert os.path.isdir(_ROOT_DIR), "Root dir: %s does not exist" % _ROOT_DIR


def rootDir():
    return _ROOT_DIR


def joinRoot(*paths):
    """Given path that is relative to the root dir, returns the absolute path"""
    return os.path.join(_ROOT_DIR, *paths)


def ensureRootPath(*paths):
    fullPath = joinRoot(*paths)
    if not os.path.exists(fullPath):
        os.makedirs(fullPath)

    return fullPath


def loadJson(path):
    with open(path, "r") as f:
        return json.load(f)


def jsonConfig(fileName):
    return loadJson(joinRoot(os.path.join("conf/common", fileName)))


def absoluteFilePaths(dirPath, ext=None, limit=float("inf")):
    """Returns a generator for absolute file paths found in a directory
    (non-recursive). Optionally filters for files with specified extension,
    e.g., 'csv' assuming a '.' delimits the extension, e.g., 'foobar.csv'"""
    c = 0
    ext = ext.lower() if ext else ext
    for root, _, files in os.walk(dirPath):
        for fname in files:
            if not ext or fname.split(".")[-1].lower() == ext:
                c += 1
                yield os.path.join(root, fname)

            if c == limit:
                break
