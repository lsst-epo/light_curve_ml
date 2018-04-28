import os
import tarfile

from lcml.utils.context_util import joinRoot, absoluteFilePaths


def getDatasetFilePaths(datasetName, ext):
    """Returns the full paths of all dataset files in project data directory:
    ./light_curve_ml/data/
    :param datasetName - Name of specific data whose individual file paths will
    be returned
    :param ext - Required file extension of dataset files
    """
    path = joinRoot("data", datasetName)
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def unarchiveAll(directory, ext="tar", mode="r:", remove=False):
    """Given a directory, untars all tar files found to that same dir.
    Optionally specify archive extension, compression type, and whether to
    remove archive file after unarchiving."""
    for i, f in enumerate(absoluteFilePaths(directory, ext=ext)):
        with tarfile.open(f, mode) as tar:
            tar.extractall(path=directory)

        if remove:
            os.remove(f)


def ensurePath(p):
    """Given a full path, ensures the directory structure for that path exists.
    E.g., given '/a/b/c.txt' generates /a/b
    """
    ensureDirs(os.path.dirname(p))


def ensureDirs(p):
    """Given a directory path, ensures the directory structure exists.
    E.g., given '/a/b/' generates /a/b"""
    if not os.path.exists(p):
        os.makedirs(p)


if __name__ == "__main__":
    ensurePath("/Users/ryanjmccall/Desktop/foobar/foobar.txt")
