from __future__ import division
from multiprocessing import cpu_count, Pool

from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


def mapMultiprocess(func, jobArgs, reportFrequency=100):
    """Executes a function on a batch of inputs using multiprocessing in an
    unordered fashion (`multiprocessing.Pool.imap_unordered`). Reports progress
    periodically as jobs complete

    :param func: function to execute
    :param jobArgs: list of tuples where each tuple is the arguments to `func`
    for a single job
    :param reportFrequency: After a batch of jobs having this size completes,
    log simple status report"""
    results = []
    pool = Pool(processes=cpu_count())
    jobs = len(jobArgs)
    for i, result in enumerate(pool.imap_unordered(func, jobArgs), 1):
        results.append(result)
        if not i % reportFrequency:
            logger.info("progress: {0:.3%} count: {1:,d}".format(i / jobs, i))

    return results


def feetsExtract(args):
    """Function to execute the feets library's feature extraction using
    multiprocesing"""
    return _feetsExtract(*args)


def _feetsExtract(featureSpace, category, timeData, magnitudeData, errorData):
    _, values = featureSpace.extract(timeData, magnitudeData, errorData)
    return values, category
