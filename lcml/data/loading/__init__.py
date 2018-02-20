"""These functions are duck-typed for input: dataDir: str, limit: int and
outputs: labels: List[str], times: List[ndarray], magnitudes: List[ndarray],
errors: List[ndarray]"""
import numpy as np

from lcml.utils.basic_logging import BasicLogging
from lcml.utils.context_util import absoluteFilePaths


logger = BasicLogging.getLogger(__name__)


def loadOgle3Dataset(dataDir, limit):
    """Loads all OGLE3 data files from specified directory as light curves
    represented as lists of the following values: classLabels, times,
    magnitudes, and magnitude errors. Class labels are parsed from originating
    data file name."""
    labels = list()
    times = list()
    magnitudes = list()
    errors = list()
    paths = absoluteFilePaths(dataDir, ext="dat", limit=limit)
    if not paths:
        raise ValueError("No data files found in %s with ext dat" % dataDir)

    for i, f in enumerate(paths):
        fileName = f.split("/")[-1]
        fnSplits = fileName.split("-")
        if len(fnSplits) > 2:
            category = fnSplits[2].lower()
        else:
            logger.warning("file name lacks category! %s", fileName)
            continue

        lc = np.loadtxt(f)
        if lc.ndim == 1:
            lc.shape = 1, 3

        labels.append(category)
        times.append(lc[:, 0])
        magnitudes.append(lc[:, 1])
        errors.append(lc[:, 2])

    return labels, times, magnitudes, errors

def loadK2Dataset(dataDir, limit):
    """Parses Light curves from LSST csv K2 data. Ignore data having
    nonzero SAP_QUALITY.

    Col 0 - TIME [64-bit floating point] - The time at the mid-point of the
    cadence in BKJD. Kepler Barycentric Julian Day (BKJD) is Julian day minus
    2454833.0 (UTC=January 1, 2009 12:00:00) and corrected to be the arrival
    times at the barycenter of the Solar System.

    Col 7 - PDCSAP_FLUX [32-bit floating point] - The flux contained in the
    optimal aperture in electrons per second after the PDC module has applied
    its cotrending algorithm to the PA light curve. To better understand how
    PDC manipulated the light curve, read Section 2.3.1.2 and see the PDCSAPFL
    keyword in the header.

    Col 8 - PDCSAP_FLUX_ERR [32-bit floating point] - The 1-sigma error in PDC
    flux values.

    Col 9 - SAP_QUALITY [32-bit integer] - Flags containing information about
    the quality of the data. Table 2-3 explains the meaning of each active bit.
    See the Data Characteristics Handbook and Data Release Notes for more
    details on safe modes, coarse point, argabrightenings, attitude tweaks, etc.
    Unused bits are reserved for future use.

    :param dataDir:
    :param limit:
    :return:
    """
    labels = list()
    times = list()
    magnitudes = list()
    errors = list()

    data = np.genfromtxt(dataDir, delimiter=",", dtype=None)
    times = data[1:][0]
    magnitudes = data[1:][7]
    errors = data[1:][8]
    flags = data[1:][9]
    # TODO check flags and obtain class labels

    return labels, times, magnitudes, errors


if __name__ == "__main__":
    _path = "/Users/ryanjmccall/code/light_curve_ml/data/k2/k2-sample.csv"
    _limit = 10
    data = loadK2Dataset(_path, _limit)
    print(len(data[0]))
