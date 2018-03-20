from lcml.pipeline.ml_pipeline_conf import loadPipelineConf
from lcml.pipeline.supervised_pipeline import SupervisedPipeline
from lcml.pipeline.unsupervised_pipeline import UnsupervisedPipeline
from lcml.utils.context_util import joinRoot, loadJson


def fromRelativePath(relPath):
    """Constructs a pipeline from config found at relative path. Relative config
    overwrites general config found at `$LSST/conf/common/pipeline.json`

    :param relPath: rel path to specific config overriding default config
    :return: Instance of `lcml.pipeline.batch_pipeline.BatchPipeline`
    """
    conf = loadJson(joinRoot("conf/common/pipeline.json"))
    relConf = loadJson(joinRoot(relPath))
    conf.update(relConf)
    pipeConf = loadPipelineConf(conf)
    pipeType = pipeConf.globalParams["type"]
    if pipeType == "supervised":
        pipe = SupervisedPipeline(pipeConf)
    elif pipeType == "unsupervised":
        pipe = UnsupervisedPipeline(pipeConf)
    else:
        raise ValueError("unsupported pipe type: %s" % pipeType)

    return pipe
