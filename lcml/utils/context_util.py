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


def absoluteFilePaths(directory, ext=None, limit=float("inf")):
    """Returns all absolute files paths found in a directory (non-recursive).
    Can optionally filter for files with specified ext, e.g. 'csv' assuming a
    '.' delimits the extension, 'foobar.csv'"""

    paths = []
    for filename in os.listdir(directory):
        if not ext or filename.split(".")[-1].lower() == ext:
            paths.append(os.path.abspath(os.path.join(directory, filename)))
            if len(paths) >= limit:
                break

    return paths
