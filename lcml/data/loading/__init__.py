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


def loadMachoDataset(dataPath, limit):
    """Loads MACHO data having red and blue light curve bands. The different
    bands are simply treated as different light curves. The returned arrays
    have the red band in the first half and the blue band in the second.

    Source data header:
    blank_ignore,field_id,tile_id,star_sequence_id,observation_date,
    observation_id,side_of_pier,exposure_time,airmass,red_magnitude,red_error,
    red_normsky,red_type,red_crowd,red_chi2,red_mpix,red_cosmicrf,red_amp,
    red_xpix,red_ypix,red_avesky,red_fwhm,red_tobs,red_cut,blue_magnitude,
    blue_error,blue_normsky,blue_type,blue_crowd,blue_chi2,blue_mpix,
    blue_cosmicrf,blue_amp,blue_xpix,blue_ypix,blue_avesky,blue_fwhm,
    blue_tobs,blue_cut
    """
    # Desired data columns:
    # ?-class, 1-field_id, 2-tile_id, 3-star_sequence_id, 4-observation_date,
    # 5-observation_id, 9-red_magnitude, 10-red_error, 24-blue_magnitude,
    # 25-blue_error
    data = np.genfromtxt(dataPath, delimiter=",", skip_header=1)[:limit]

    # double up these since we have two bands
    labels = np.tile(data[:, 0], 2)
    times = np.tile(data[:, 4], 2)
    magnitudes = data[:, 9] + data[:, 24]
    errors = data[:, 10] + data[:, 25]
    return labels, times, magnitudes, errors


def loadK2Dataset(dataPath, limit):
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

    :param dataPath: full path to source file(s)
    :param limit: restriction number of light curves returned
    :return:
    """
    # TODO can we obtain labels for k2?
    labels = list()
    data = np.genfromtxt(dataPath, delimiter=",", dtype=float, skip_header=1)
    flags = data[:, 9]

    # only select data where SAP quality flags are 0
    goodRows = np.where(flags == 0)[0][:limit]
    times = data[goodRows, 0]
    magnitudes = data[goodRows, 7]
    errors = data[goodRows, 8]

    return labels, times, magnitudes, errors


if __name__ == "__main__":
    _path = "/Users/ryanjmccall/code/light_curve_ml/data/k2/k2-sample.csv"
    _limit = 10
    _data = loadK2Dataset(_path, _limit)
    print(len(_data[1]))
