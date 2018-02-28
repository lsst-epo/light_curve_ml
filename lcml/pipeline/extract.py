from feets import FeatureSpace

from lcml.pipeline.data_format import STANDARD_INPUT_DATA_TYPES
from lcml.pipeline.preprocess import allFinite, NON_FINITE_VALUES
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.format_util import fmtPct
from lcml.utils.multiprocess import feetsExtract, mapMultiprocess


logger = BasicLogging.getLogger(__name__)


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
    logger.info("Extracting features for {:,d} LCs...".format(len(labels)))

    fs = FeatureSpace(data=STANDARD_INPUT_DATA_TYPES, exclude=exclude)
    jobs = [(fs, labels[i], times[i], mags[i], errors[i])
            for i in range(len(labels))]
    featuresAndLabels = mapMultiprocess(feetsExtract, jobs)
    validFeatures = []
    validLabels = []
    badCount = 0

    impute = params.get("impute", True)
    for features, label in featuresAndLabels:
        if allFinite(features):
            validFeatures.append(features)
            validLabels.append(label)
        else:
            if impute:
                for i, f in enumerate(features):
                    if f in NON_FINITE_VALUES:
                        # set non-finite feature values to 0.0
                        logger.warning("imputing feature: %s value: %s", i, f)
                        features[i] = 0.0

            badCount += 1

    if badCount:
        logger.warning("Bad feature value rate: %s",
                       fmtPct(badCount, len(featuresAndLabels)))

    return validFeatures, validLabels
