from lcml.pipeline.ml_pipeline_conf import loadPipelineConf
from lcml.pipeline.supervised_pipeline import SupervisedPipeline
from lcml.pipeline.unsupervised_pipeline import UnsupervisedPipeline
from lcml.utils.context_util import joinRoot, loadJson


_DEFAULT_PIPE_CONF_REL_PATH = "conf/common/pipeline.json"


def recursiveMerge(a, b):
    """Merges dicts b into a recursively overwriting existing keys"""
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                recursiveMerge(a[key], b[key])
            elif a[key] != b[key]:
                a[key] = b[key]
        else:
            a[key] = b[key]

    return a


def fromRelativePath(relPath):
    """Constructs a pipeline from config found at relative path. Relative config
    overwrites general config found at `$LCML/conf/common/pipeline.json`

    :param relPath: rel path to specific config overriding default config
    :return: Instance of `lcml.pipeline.batch_pipeline.BatchPipeline`
    """
    defaultConf = loadJson(joinRoot(_DEFAULT_PIPE_CONF_REL_PATH))
    relConf = loadJson(joinRoot(relPath))
    conf = defaultConf.copy()
    conf = recursiveMerge(conf, relConf)

    pipeConf = loadPipelineConf(conf)
    pipeType = pipeConf.globalParams["type"]
    if pipeType == "supervised":
        pipe = SupervisedPipeline(pipeConf)
    elif pipeType == "unsupervised":
        pipe = UnsupervisedPipeline(pipeConf)
    else:
        raise ValueError("unsupported pipeline type: %s" % pipeType)

    return pipe
