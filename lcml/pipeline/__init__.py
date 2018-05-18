from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.ml_pipeline_conf import GLOBAL_PARAMS, loadPipelineConf
from lcml.pipeline.supervised_pipeline import SupervisedPipeline
from lcml.pipeline.unsupervised_pipeline import UnsupervisedPipeline
from lcml.utils.context_util import joinRoot, loadJson
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


_DEFAULT_PIPE_CONF_REL_PATH = "conf/common/pipeline.json"


def recursiveMerge(a: dict, b: dict) -> dict:
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


def fromRelativePath(relPath: str) -> BatchPipeline:
    """Constructs a pipeline from config found at relative path. Relative config
    overwrites general config found at `$LCML/conf/common/pipeline.json`

    :param relPath: relative path to specific config overriding default config
    :return: constructed BatchPipeline object
    """
    defaultConf = loadJson(joinRoot(_DEFAULT_PIPE_CONF_REL_PATH))
    relConf = loadJson(joinRoot(relPath))
    conf = recursiveMerge(defaultConf.copy(), relConf)
    logger.info("Global params: %s", conf[GLOBAL_PARAMS])
    pipeConf = loadPipelineConf(conf)
    pipeType = pipeConf.globalParams["type"]
    if pipeType == "supervised":
        pipe = SupervisedPipeline(pipeConf)
    elif pipeType == "unsupervised":
        pipe = UnsupervisedPipeline(pipeConf)
    else:
        raise ValueError("unsupported pipeline type: %s" % pipeType)

    return pipe
