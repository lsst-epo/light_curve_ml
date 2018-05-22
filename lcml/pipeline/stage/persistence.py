import json
import logging
import os
import platform
import sys
from typing import Tuple, Union

import numpy as np
import sklearn
from sklearn.externals import joblib

from lcml.pipeline.stage.model_selection import (ClassificationMetrics,
                                                 ModelSelectionResult)


logger = logging.getLogger(__name__)


META_ARCH_BITS = "archBits"
META_SKLEARN = "sklearnVersion"
META_MAIN = "mainFile"
META_PIPELINE_PARAMS = "pipelineParams"
META_MODEL_HYPERPARAMS = "hyperparameters"
META_CV_METRICS = "cvMetrics"
META_TEST_METRICS = "testMetrics"
_META_FILENAME = "metadata.json"


def serPipelineResults(conf, classMapping: dict,
                       result: ModelSelectionResult,
                       testMetrics: ClassificationMetrics):
    """Save the key aspects and results of a pipeline run to disk.

    :param conf: ML pipeline conf containing params
    :param classMapping: class label mapping, int to string
    :param result: ModelSelectionResult of best model found in cross-validation
    :param testMetrics: best model's scores on test set
    """
    path = conf.serStage.params["modelSavePath"]
    if not path or not result:
        return

    if result.model:
        joblib.dump(result.model, path)
        logger.info("Saved model to: %s", path)

    archBits = platform.architecture()[0]
    mainFile = sys.modules["__main__"].__file__
    searchParams = conf.searchStage.params.copy()
    searchParams.pop("model", None)  # remove class object
    params = {META_MODEL_HYPERPARAMS: result.hyperparameters,
              "loadParams": conf.loadStage.params,
              "extractParams": conf.extractStage.params,
              "searchParams": searchParams,
              "mapping": classMapping}
    cvMetricsDict = _metricsToDict(result.metrics)
    testMetricsDict = _metricsToDict(testMetrics)
    metadata = {META_ARCH_BITS: archBits, META_SKLEARN: sklearn.__version__,
                META_MAIN: mainFile, META_PIPELINE_PARAMS: params,
                META_CV_METRICS: cvMetricsDict,
                META_TEST_METRICS: testMetricsDict}
    metadataPath = _metadataPath(path)
    with open(metadataPath, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
    logger.info("Saved metadata to: %s", metadataPath)


def _metricsToDict(metrics: Union[ClassificationMetrics, dict]) -> dict:
    if metrics is None:
        return dict()
    if isinstance(metrics, ClassificationMetrics):
        metrics = vars(metrics)
    return {k: v.tolist() if type(v) is np.ndarray else v
            for k, v in metrics.items()}


def loadModelAndHyperparms(modelPath) -> Tuple[Union[object, None],
                                               Union[dict, None]]:
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

    if model is not None and metadata is not None:
        hyperparams = metadata[META_PIPELINE_PARAMS][META_MODEL_HYPERPARAMS]
        return model, hyperparams

    return None, None


def _metadataPath(modelPath):
    finalDirLoc = modelPath.rfind(os.sep)
    return os.path.join(modelPath[:finalDirLoc], _META_FILENAME)
