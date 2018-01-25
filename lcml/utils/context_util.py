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



def absoluteFilePaths(directory, ext=None):
    """Returns all absolute files paths found in a directory (non-recursive).
    Can optionally filter for files with specified ext, e.g. 'csv' assuming a
    '.' delimits the extension, 'foobar.csv'"""
    if ext:
        ext = ext.lower()
        return [os.path.abspath(os.path.join(dirPath, f))
                for dirPath, _, fileNames in os.walk(directory)
                for f in fileNames
                if f.split(".")[-1].lower() == ext]
    else:
        return [os.path.abspath(os.path.join(dirPath, f))
                for dirPath, _, fileNames in os.walk(directory)
                for f in fileNames]
