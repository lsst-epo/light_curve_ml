#!/usr/bin/env python3
import multiprocessing
from multiprocessing import Pool
import time


def workFunc(taskQueue, resultQueue, *args):
    while True:
        nextTask = taskQueue.get()
        if nextTask is None:
            break

        resultQueue.put(nextTask["v"] * args[0])


def multiprocessingQueueTest(problems):
    # Establish communication queues
    taskQueue = multiprocessing.JoinableQueue()
    resultQueue = multiprocessing.Queue()

    for p in range(problems):
        taskQueue.put({"v": p})

    numCPU = multiprocessing.cpu_count()
    for _ in range(numCPU):
        taskQueue.put(None)

    a = 2
    jobs = []
    for i in range(numCPU):
        p = multiprocessing.Process(target=workFunc,
                                    args=(taskQueue, resultQueue, a))
        jobs.append(p)
        p.daemon = True
        p.start()

    # taskQueue.join()
    while not taskQueue.empty():
        print("waiting...")
        time.sleep(1)

    sum = 0
    while not resultQueue.empty():
        sum += resultQueue.get()

    print("sum %s" % sum)


def f(a):
    return 2 * a


def main(size):
    pool = Pool(processes=6) # run no more than 6 at a time

    s = time.time()
    result = pool.map(f, range(size)) # pass full list (12 items)
    total = sum(result)
    print("%s in %.2fs" % (total, time.time() - s))


if __name__ == "__main__":
    main(int(1e5))
