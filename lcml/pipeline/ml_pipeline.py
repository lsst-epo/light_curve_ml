from collections import namedtuple

from lcml.data import loading
from lcml.pipeline.extract import feetsExtractFeatures
from lcml.pipeline.model_selection import gridSearchSelection
from lcml.utils.context_util import joinRoot, loadJson


#: global parameters section of the pipeline conf file
GLOBAL_PARAMS = "globalParams"
DB_PARAMS = "database"
LOAD_DATA = "loadData"
EXTRACT_FEATURES = "extractFeatures"
MODEL_SELECTION = "modelSelection"
SERIALIZATION = "serialization"


#: Some pipeline components consist of a function and parameters
FunctionAndParams = namedtuple("FunctionAndParams", ["fcn", "params"])


class MlPipeline:
    """Container for functions and parameter of the major components of a ML
     pipeline"""
    def __init__(self, globalParams, dbParams, loadData, extractFeatures,
                 modelSelection, serialParams):
        self.globalParams = globalParams
        self.dbParams = dbParams
        self.loadData = loadData
        self.extractFeatures = extractFeatures
        self.modelSelection = modelSelection
        self.serialParams = serialParams


def fromRelativePath(relPath):
    path = joinRoot(relPath)
    return loadPipeline(loadJson(path))


def loadPipeline(conf):
    """Constructs a pipeline from a .json config."""
    # load data fcn
    loadType = conf[LOAD_DATA]["function"].lower()
    if loadType == "ogle3":
        loadFcn = loading.loadOgle3Dataset
    elif loadType == "macho":
        loadFcn = loading.loadMachoDataset
    elif loadType == "k2":
        loadFcn = loading.loadK2Dataset
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
                      dbParams=conf[DB_PARAMS],
                      loadData=FunctionAndParams(loadFcn, loadParams),
                      extractFeatures=FunctionAndParams(extractFcn, extParams),
                      modelSelection=FunctionAndParams(selectFcn, selParams),
                      serialParams=conf[SERIALIZATION]["params"])
