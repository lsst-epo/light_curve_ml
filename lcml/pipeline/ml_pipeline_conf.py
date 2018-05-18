from collections import namedtuple
import importlib

from lcml.data.loading.csv_file_loading import loadFlatLcDataset
from lcml.pipeline.stage.extract import feetsExtractFeatures
from lcml.pipeline.stage.model_selection import gridSearchCv
from lcml.pipeline.stage.persistence import serPipelineResults
from lcml.pipeline.stage.postprocess import postprocessFeatures
from lcml.pipeline.stage.preprocess import cleanLightCurves
from lcml.utils.pathing import ensurePath


#: global parameters used throughout pipeline stages
GLOBAL_PARAMS = "globalParams"


#: pipeline's database and tables
DB_PARAMS = "database"


#: stage moving data from original format (e.g., csv) to db table
LOAD_DATA_STAGE = "loadData"


#: light curve preprocessing
PREPROCESS_DATA_STAGE = "preprocessData"


#: stage processing clean data into feature vectors
EXTRACT_FEATURES_STAGE = "extractFeatures"


#: additional processing after feature extraction
POST_PROC_FEATS_STAGE = "postprocessFeatures"


#: model search and selection
MODEL_SEARCH_STAGE = "modelSearch"


#: persisting pipeline results to disk
SERIALIZATION = "serialization"


#: Major pipeline processing stage involving data processing writing results to
#: table
PipelineStage = namedtuple("PipelineStage", ["skip", "fcn", "params",
                                             "writeTable"])


# Candidate for data class
class MlPipelineConf:
    """Container for functions and parameter of the major components of a ML
     pipeline"""
    def __init__(self, globalParams: dict, dbParams: dict,
                 loadStage: PipelineStage, preprocessStage: PipelineStage,
                 extractStage: PipelineStage, ftProcessStage: PipelineStage,
                 searchStage: PipelineStage, serStage: PipelineStage):
        self.globalParams = globalParams
        self.dbParams = dbParams
        self.loadStage = loadStage
        self.preprocessStage = preprocessStage
        self.extractStage = extractStage
        self.postprocessStage = ftProcessStage
        self.searchStage = searchStage
        self.serStage = serStage


def _makeInstance(modelClass: str, params: dict) -> object:
    strInd = modelClass.rfind(".")
    moduleName = modelClass[:strInd]
    className = modelClass[strInd + 1:]
    module = importlib.import_module(moduleName)
    class_ = getattr(module, className)
    return class_(**params)


def loadPipelineConf(conf: dict) -> MlPipelineConf:
    """Constructs a pipeline from a .json config."""
    dbParams = conf[DB_PARAMS]
    ensurePath(dbParams["dbPath"])

    # Stage: Load Data
    loadStage = _loadStage(conf[LOAD_DATA_STAGE], dbParams, loadFlatLcDataset)

    # Stage: Clean Data
    preprocStage = _loadStage(conf[PREPROCESS_DATA_STAGE], dbParams,
                              cleanLightCurves)

    # Stage: Extract Features
    extractType = conf[EXTRACT_FEATURES_STAGE]["function"]
    if extractType == "feets":
        extFcn = feetsExtractFeatures
    else:
        raise ValueError("unsupported extract function: %s" % extractType)
    extractStage = _loadStage(conf[EXTRACT_FEATURES_STAGE], dbParams, extFcn)

    # Stage: Postprocess features
    postprocessStage = _loadStage(conf[POST_PROC_FEATS_STAGE], dbParams,
                                  postprocessFeatures)

    # Stage: Model Search
    stgCnf = conf[MODEL_SEARCH_STAGE]
    searchType = stgCnf["function"]
    if searchType == "grid":
        searchFcn = gridSearchCv
    else:
        raise ValueError("unsupported search function: %s" % searchType)

    searchParams = stgCnf["params"]
    if "model" in stgCnf:
        _params = stgCnf["model"]["params"]
        _params["random_state"] = conf[GLOBAL_PARAMS]["randomState"]
        searchParams["model"] = _makeInstance(stgCnf["model"]["class"],
                                              _params)
    else:
        searchParams["model"] = None

    searchStage = _loadStage(conf[MODEL_SEARCH_STAGE], dbParams, searchFcn)

    # Stage: Pipeline result serialization
    stgCnf = conf[SERIALIZATION]
    ensurePath(stgCnf["params"]["modelSavePath"])
    serialStage = _loadStage(stgCnf, dbParams, serPipelineResults)
    return MlPipelineConf(conf[GLOBAL_PARAMS], conf[DB_PARAMS], loadStage,
                          preprocStage, extractStage, postprocessStage,
                          searchStage, serialStage)


def _loadStage(stageConf: dict, dbParams: dict, fcn) -> PipelineStage:
    writeTable = stageConf.get("writeTable", None)
    if writeTable:
        # retrieve actual db name from db table definition
        writeTable = dbParams[writeTable]

    return PipelineStage(stageConf.get("skip", None),
                         fcn,
                         stageConf["params"],
                         writeTable)
