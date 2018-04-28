import json
import os

from typing import Generator


_ROOT_ENV_VAR = "LCML"
_ROOT_DIR = os.environ.get(_ROOT_ENV_VAR, None)
assert _ROOT_DIR is not None, ("Please set the '%s' environment variable" %
                               _ROOT_ENV_VAR)
assert os.path.isdir(_ROOT_DIR), "Root dir: %s does not exist" % _ROOT_DIR


def rootDir() -> str:
    return _ROOT_DIR


def joinRoot(*paths) -> str:
    """Given path that is relative to the root dir, returns the absolute path"""
    return os.path.join(_ROOT_DIR, *paths)


def loadJson(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def jsonConfig(fileName: str) -> dict:
    return loadJson(joinRoot(os.path.join("conf/common", fileName)))


def absoluteFilePaths(dirPath: str,
                      ext: str=None,
                      limit=float("inf")) -> Generator[str, None, None]:
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
