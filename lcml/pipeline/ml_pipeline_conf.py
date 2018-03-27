from collections import namedtuple
import importlib

from lcml.data.loading.csv_file_loading import loadFlatLcDataset
from lcml.pipeline.stage.extract import feetsExtractFeatures
from lcml.pipeline.stage.model_selection import randomForestGridSearch
from lcml.utils.pathing import ensureDir


#: global parameters section of the pipeline conf file
GLOBAL_PARAMS = "globalParams"
DB_PARAMS = "database"
LOAD_DATA = "loadData"
EXTRACT_FEATURES = "extractFeatures"
MODEL_SEARCH = "modelSearch"
SERIALIZATION = "serialization"


#: Some pipeline components consist of a function and parameters
FunctionAndParams = namedtuple("FunctionAndParams", ["fcn", "params"])


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


def loadPipelineConf(conf):
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

    ensureDir(conf[DB_PARAMS]["dbPath"])
    extParams = conf[EXTRACT_FEATURES]["params"]
    extractFeatures = FunctionAndParams(extractFcn, extParams)

    searchType = conf[MODEL_SEARCH]["function"]
    if searchType == "grid":
        searchFcn = randomForestGridSearch
    else:
        raise ValueError("unsupported search function: %s" % searchType)

    searchParams = conf[MODEL_SEARCH]["params"]

    modelType = conf[MODEL_SEARCH]["params"]["model"]
    strInd = modelType.rfind(".")
    moduleName = modelType[:strInd]
    className = modelType[strInd + 1:]
    module = importlib.import_module(moduleName)
    class_ = getattr(module, className)
    searchParams["model"] = class_

    ensureDir(conf[SERIALIZATION]["params"]["modelSavePath"])
    modelSearch = FunctionAndParams(searchFcn, searchParams)

    serialParams = conf[SERIALIZATION]["params"]
    return MlPipelineConf(conf[GLOBAL_PARAMS], conf[DB_PARAMS], loadData,
                          extractFeatures, modelSearch, serialParams)
