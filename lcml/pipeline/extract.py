
from feets import FeatureSpace
import numpy as np

from lcml.pipeline.data_format import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.preprocess import allFinite
from lcml.utils.basic_logging import getBasicLogger
from lcml.utils.format_util import fmtPct
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess



logger = getBasicLogger(__name__, __file__)


_EXPENSIVE_FEATS = ["CAR_mean", "CAR_sigma", "CAR_tau"]


def feetsExtractFeatures(labels, times, mags, errors, params):
    """Runs light curves through 'feets' library obtaining feature vectors.
    Perfoms the extraction using multiprocessing. Output order will not
    necessarily correspond to input order, therefore, class labels are returned
    as well aligned with feature vectors to avoid confusion.

    :param labels: light curve class labels
    :param times: light curve times
    :param mags: light curve magnitudes
    :param errors: light curve magnitude errors
    :param params: optional parameters
    :returns feature vectors for each LC and list of corresponding class labels
    """
    exclude = (list if params.get("allFeatures", False) else _EXPENSIVE_FEATS)
    logger.info("Excluded features: %s", exclude)
    logger.info("Extracting features...")

    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    cleanLcDf = [(fs, labels[i], times[i], mags[i], errors[i])
                 for i in range(len(labels))]
    featureLabels, _ = mapMultiprocess(feetsExtract, cleanLcDf)
    validFeatures = []
    validLabels = []
    badCount = 0

    # if True, set all extracted features having value 'nan' to 0.0
    impute = params.get("impute", True)
    for features, label in featureLabels:
        if allFinite(features):
            validFeatures.append(features)
            validLabels.append(label)
        else:
            if impute:
                for i, f in enumerate(features):
                    if np.isnan(f):
                        features[i] = 0.0

            logger.warning("bad feature set: %s", features)
            badCount += 1

    if badCount:
        logger.warning("Skipped b/c nan rate: %s", fmtPct(badCount,
                                                          len(featureLabels)))

    return validFeatures, validLabels

