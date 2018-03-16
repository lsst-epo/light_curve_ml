from collections import namedtuple

from lcml.data.loading.csv_file_loading import loadFlatLcDataset
from lcml.pipeline.stage.extract import feetsExtractFeatures
from lcml.pipeline.stage.model_selection import gridSearchSelection
from lcml.utils.pathing import ensureDir


#: global parameters section of the pipeline conf file
GLOBAL_PARAMS = "globalParams"
DB_PARAMS = "database"
LOAD_DATA = "loadData"
EXTRACT_FEATURES = "extractFeatures"
MODEL_SELECTION = "modelSelection"
SERIALIZATION = "serialization"


#: Some pipeline components consist of a function and parameters
FunctionAndParams = namedtuple("FunctionAndParams", ["fcn", "params"])


class MlPipelineConf:
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

    selectionType = conf[MODEL_SELECTION]["function"]
    if selectionType == "grid":
        selectFcn = gridSearchSelection
    else:
        raise ValueError("unsupported selection function: %s" % selectionType)

    selectParams = conf[MODEL_SELECTION]["params"]
    ensureDir(conf[SERIALIZATION]["params"]["modelSavePath"])
    modelSelection = FunctionAndParams(selectFcn, selectParams)
    serialParams = conf[SERIALIZATION]["params"]
    return MlPipelineConf(conf[GLOBAL_PARAMS], conf[DB_PARAMS], loadData,
                          extractFeatures, modelSelection, serialParams)
