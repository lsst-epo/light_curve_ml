from multiprocessing import cpu_count, Pool
import time


def mapMultiprocess(func, lcs):
    pool = Pool(processes=cpu_count())
    start = time.time()
    results = pool.map(func, lcs)
    minElapsed = (time.time() - start) / 60
    return results, minElapsed


def feetsExtract(args):
    return _feetsExtract(*args)


def _feetsExtract(featureSpace, category, timeData, magnitudeData, errorData):
    _, values = featureSpace.extract(timeData, magnitudeData, errorData)
    return values, category
