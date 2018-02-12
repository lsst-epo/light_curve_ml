from collections import namedtuple
import json

from lcml.data.loading import loadOgle3Dataset
from lcml.utils.context_util import joinRoot


class MlPipeline:
    def __init__(self, loadData, extractFeatures, modelSelection):
        self.loadData = loadData
        self.extractFeatures = extractFeatures
        self.modelSelection = modelSelection


def fromRelativePath(relPath):
    path = joinRoot(relPath)
    return loadPipeline(loadJson(path))


def loadJson(path):
    with open(path, "r") as f:
        return json.load(f)


FunctionAndArgs = namedtuple("FunctionAndArgs", ["fcn", "args"])


def loadPipeline(conf):
    """Constructs a pipeline from a json config"""
    # load data fcn
    if conf["loadData"]["function"] == "ogle3":
        loadFcn = loadOgle3Dataset
    else:
        raise ValueError("unsupported loadDataFcn %s" % conf["loadDataFcn"])

    loadParams = conf["loadData"]["params"]
    extractFcn = conf["extractFeatures"]["function"]
    modelSelectionFcn = conf["modelSelection"]["method"]
    modelSelectionParams = conf["modelSelection"]["params"]
    return MlPipeline(loadData=FunctionAndArgs(loadFcn, loadParams),
                      extractFeatures=FunctionAndArgs(extractFcn, None),
                      modelSelection=FunctionAndArgs(modelSelectionFcn,
                                                     modelSelectionParams))
