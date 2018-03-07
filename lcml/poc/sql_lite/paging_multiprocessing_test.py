import time
from multiprocessing import cpu_count, Pool
import sqlite3


def work(columnValue):
    return len(list(range(1000000)))


def workGenerator():
    path = "/Users/ryanjmccall/code/light_curve_ml/data/ogle3/ogle3_processed.db"
    table = "clean_lcs"
    column = "id"
    lastValue = ""
    pageSize = 100
    qBase = ("SELECT * FROM {0} "
             "WHERE {1} > \"{2}\" "
             "ORDER BY {1} "
             "LIMIT {3}")

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    rows = True
    while rows:
        q = qBase.format(table, column, lastValue, pageSize)
        cursor.execute(q)
        rows = cursor.fetchall()
        for r in rows:
            yield r[0]

        if rows:
            lastValue = rows[-1][0]


def main():
    start = time.time()
    reportFrequency = 100
    pool = Pool(processes=cpu_count())
    results = []
    for i, result in enumerate(pool.imap_unordered(work, workGenerator()), 1):
        results.append(result)
        if i % reportFrequency == 0:
            print("progress: {0:,d} sample result: {1}".format(i, result))

    print("result count: %s" % len(results))
    print("elapsed: %s" % (time.time() - start))
    # 1 core took 16.5s
    # 4 cores took 9s


if __name__ == "__main__":
    main()
