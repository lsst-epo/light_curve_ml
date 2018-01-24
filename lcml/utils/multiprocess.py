from multiprocessing import cpu_count, Pool
import time


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
    except BaseException as e:
        print(e)
        values = None

    return values, category
