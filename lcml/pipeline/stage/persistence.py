import json
import os
import platform
import sys

import numpy as np
import sklearn
from sklearn.externals import joblib

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


META_ARCH_BITS = "archBits"
META_SKLEARN = "sklearnVersion"
META_MAIN = "mainFile"
META_PARAMS = "params"
META_PARAMS_HYPER = "hyperparameters"
META_METRICS = "metrics"
_META_FILENAME = "metadata.json"


def saveModel(result, modelPath, pipe, classToLabel):
    """If 'modelPath' is specified, the model and its metadata, including
    'trainParams' and 'cvScore' are saved to disk.

    :param result: ModelSelectionResult to save
    :param modelPath: save path
    :param pipe: ML pipeline
    :param classToLabel: mapping from int to class label
    """
    searchParams = pipe.modelSearch.params
    searchParams.pop("model", None)  # remove class object
    params = {META_PARAMS_HYPER: result.hyperparameters,
              "loadParams": pipe.loadData.params,
              "extractParams": pipe.extractFeatures.params,
              "searchParams": searchParams}
    metrics = vars(result.metrics)
    metrics["mapping"] = classToLabel

    joblib.dump(result.model, modelPath)
    logger.info("Dumped model to: %s", modelPath)
    metadataPath = _metadataPath(modelPath)
    archBits = platform.architecture()[0]
    mainFile = sys.modules["__main__"].__file__
    metricsJson = {k: v.tolist() if type(v) is np.ndarray else v
                   for k, v in metrics.items()}
    metadata = {META_ARCH_BITS: archBits, META_SKLEARN: sklearn.__version__,
                META_MAIN: mainFile, META_PARAMS: params,
                META_METRICS: metricsJson}
    with open(metadataPath, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    logger.info("Wrote metadata to: %s", metadataPath)


def loadModels(modelPath):
    """Load previous winning model and its metadata from disk"""
    try:
        model = joblib.load(modelPath)
    except IOError:
        logger.warning("Failed to load model from: %s", modelPath)
        return None, None

    logger.info("Loaded model from: %s", modelPath)
    metadataPath = _metadataPath(modelPath)
    try:
        with open(metadataPath) as mFile:
            metadata = json.load(mFile)
    except IOError:
        logger.warning("Metadata file doesn't exist: %s", metadataPath)
        return None, None

    if metadata[META_ARCH_BITS] != platform.architecture()[0]:
        logger.critical("Model created on arch: %s but current arch is %s",
                        metadata[META_ARCH_BITS], platform.architecture()[0])
        raise ValueError("Unusable model")

    models = None
    if model is not None and metadata is not None:
        hyperparams = metadata[META_PARAMS][META_PARAMS_HYPER]
        models = [(model, hyperparams)]

    return models


def _metadataPath(modelPath):
    finalDirLoc = modelPath.rfind(os.sep)
    return os.path.join(modelPath[:finalDirLoc], _META_FILENAME)
