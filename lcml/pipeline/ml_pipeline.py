from collections import namedtuple

from sklearn.ensemble import RandomForestClassifier

from lcml.data.loading import loadOgle3Dataset
from lcml.pipeline.extract import feetsExtractFeatures
from lcml.utils.context_util import joinRoot, loadJson


#: global parameters section of the pipeline conf file
GLOBAL_PARAMS = "globalParams"
LOAD_DATA = "loadData"
EXTRACT_FEATURES = "extractFeatures"
MODEL_SELECTION = "modelSelection"
SERIALIZATION = "serialization"


class MlPipeline:
    def __init__(self, globalParams, loadData, extractFeatures, modelSelection,
                 serialParams):
        self.globalParams = globalParams
        self.loadData = loadData
        self.extractFeatures = extractFeatures
        self.modelSelection = modelSelection
        self.serialParams = serialParams


def fromRelativePath(relPath):
    path = joinRoot(relPath)
    return loadPipeline(loadJson(path))


FunctionAndParams = namedtuple("FunctionAndParams", ["fcn", "params"])


def gridSearchSelection(params):
    # default for num estimators is 10
    estimatorsStart = params["estimatorsStart"]
    estimatorsStop = params["estimatorsStop"]

    # default for max features is sqrt(len(features))
    # for feets len(features) ~= 64 => 8
    rfFeaturesStart = params["rfFeaturesStart"]
    rfFeaturesStop = params["rfFeaturesStop"]
    return [(RandomForestClassifier(n_estimators=t, max_features=f,
                                    n_jobs=params["jobs"]),
             {"trees": t, "maxFeatures": f})
            for f in range(rfFeaturesStart, rfFeaturesStop)
            for t in range(estimatorsStart, estimatorsStop)]


def loadPipeline(conf):
    """Constructs a pipeline from a json config"""
    # load data fcn
    loadType = conf[LOAD_DATA]["function"]
    if loadType == "ogle3":
        loadFcn = loadOgle3Dataset
    else:
        raise ValueError("unsupported load function: %s" % loadType)

    loadParams = conf[LOAD_DATA]["params"]

    extractType = conf[EXTRACT_FEATURES]["function"]
    if extractType == "feets":
        extractFcn = feetsExtractFeatures
    else:
        raise ValueError("unsupported extract function: %s" % extractType)

    extParams = conf[EXTRACT_FEATURES]["params"]

    selectionType = conf[MODEL_SELECTION]["function"]
    if selectionType == "grid":
        selectFcn = gridSearchSelection
    else:
        raise ValueError("unsupported selection function: %s" % selectionType)

    selParams = conf[MODEL_SELECTION]["params"]
    return MlPipeline(globalParams=conf[GLOBAL_PARAMS],
                      loadData=FunctionAndParams(loadFcn, loadParams),
                      extractFeatures=FunctionAndParams(extractFcn, extParams),
                      modelSelection=FunctionAndParams(selectFcn, selParams),
                      serialParams=conf[SERIALIZATION]["params"])
