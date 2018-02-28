from __future__ import division
from multiprocessing import cpu_count, Pool

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def mapMultiprocess(func, jobArgs, reportFrequency=100):
    results = []
    pool = Pool(processes=cpu_count())
    jobs = len(jobArgs)
    for i, result in enumerate(pool.imap_unordered(func, jobArgs), 1):
        results.append(result)
        if not i % reportFrequency:
            logger.info("progress: {0:.3%} count: {1:,d}".format(i / jobs, i))

    return results


def feetsExtract(args):
    return _feetsExtract(*args)


def _feetsExtract(featureSpace, category, timeData, magnitudeData, errorData):
    _, values = featureSpace.extract(timeData, magnitudeData, errorData)
    return values, category
