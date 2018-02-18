from multiprocessing import cpu_count, Pool
import time

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def mapMultiprocess(func, args):
    pool = Pool(processes=cpu_count())
    start = time.time()
    results = pool.map(func, args)
    minElapsed = (time.time() - start) / 60
    return results, minElapsed


def feetsExtract(args):
    return _feetsExtract(*args)


def _feetsExtract(featureSpace, category, timeData, magnitudeData, errorData):
    try:
        _, values = featureSpace.extract(timeData, magnitudeData, errorData)
    except BaseException:
        logger.exception("feets feature extract failed for data: time: %s"
                         " mag: %s err: %s", timeData, magnitudeData, errorData)
        values = None

    return values, category
