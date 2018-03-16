from lcml.pipeline.ml_pipeline_conf import loadPipelineConf
from lcml.pipeline.supervised_pipeline import SupervisedPipeline
from lcml.pipeline.unsupervised_pipeline import UnsupervisedPipeline
from lcml.utils.context_util import joinRoot, loadJson


def fromRelativePath(relPath):
    path = joinRoot(relPath)
    conf = loadPipelineConf(loadJson(path))

    pipeType = conf.globalParams["type"]
    if pipeType == "supervised":
        pipe = SupervisedPipeline(conf)
    elif pipeType == "unsupervised":
        pipe = UnsupervisedPipeline(conf)
    else:
        raise ValueError("unsupported pipe type: %s" % pipeType)

    return pipe
