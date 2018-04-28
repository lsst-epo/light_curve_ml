from collections import namedtuple
import importlib

from lcml.data.loading.csv_file_loading import loadFlatLcDataset
from lcml.pipeline.stage.extract import feetsExtractFeatures
from lcml.pipeline.stage.model_selection import gridSearchCv
from lcml.utils.pathing import ensurePath


#: global parameters section of the pipeline conf file
GLOBAL_PARAMS = "globalParams"
DB_PARAMS = "database"
LOAD_DATA = "loadData"
EXTRACT_FEATURES = "extractFeatures"
MODEL_SEARCH = "modelSearch"
SERIALIZATION = "serialization"


#: Some pipeline components consist of a function and parameters
FunctionAndParams = namedtuple("FunctionAndParams", ["fcn", "params"])


# Candidate for data class
class MlPipelineConf:
    """Container for functions and parameter of the major components of a ML
     pipeline"""
    def __init__(self, globalParams, dbParams, loadData, extractFeatures,
                 modelSearch, serialParams):
        self.globalParams = globalParams
        self.dbParams = dbParams
        self.loadData = loadData
        self.extractFeatures = extractFeatures
        self.modelSearch = modelSearch
        self.serialParams = serialParams


def _initModel(modelConfig: dict) -> object:
    modelClass = modelConfig["class"]
    strInd = modelClass.rfind(".")
    moduleName = modelClass[:strInd]
    className = modelClass[strInd + 1:]
    module = importlib.import_module(moduleName)
    class_ = getattr(module, className)
    return class_(**modelConfig["params"])


def loadPipelineConf(conf: dict) -> MlPipelineConf:
    """Constructs a pipeline from a .json config."""
    # load data fcn
    # loadType = conf[LOAD_DATA]["function"].lower()
    loadFcn = loadFlatLcDataset
    loadParams = conf[LOAD_DATA]["params"]
    loadData = FunctionAndParams(loadFcn, loadParams)

    extractType = conf[EXTRACT_FEATURES]["function"]
    if extractType == "feets":
        extractFcn = feetsExtractFeatures
    else:
        raise ValueError("unsupported extract function: %s" % extractType)

    ensurePath(conf[DB_PARAMS]["dbPath"])
    extParams = conf[EXTRACT_FEATURES]["params"]
    extractFeatures = FunctionAndParams(extractFcn, extParams)

    searchType = conf[MODEL_SEARCH]["function"]
    if searchType == "grid":
        searchFcn = gridSearchCv
    else:
        raise ValueError("unsupported search function: %s" % searchType)

    searchParams = conf[MODEL_SEARCH]["params"]
    if "model" in conf[MODEL_SEARCH]:
        searchParams["model"] = _initModel(conf[MODEL_SEARCH]["model"])
    else:
        searchParams["model"] = None

    modelSearch = FunctionAndParams(searchFcn, searchParams)

    serialParams = conf[SERIALIZATION]["params"]
    ensurePath(serialParams["modelSavePath"])
    return MlPipelineConf(conf[GLOBAL_PARAMS], conf[DB_PARAMS], loadData,
                          extractFeatures, modelSearch, serialParams)
