#!/usr/bin/env python3
import dask.dataframe as dd
import pandas as pd
import numpy as np

from lcml.utils.context_util import joinRoot


path = joinRoot("data/ogle3/ogle3.csv")


desiredColumns = ['OGLE3_HJD', 'OGLE3_MAGNITUDE', 'OGLE3_ERROR',
                  'OGLE3_CATEGORY',
                  'OGLE3_ID']
df = dd.read_csv(path)
# df = pd.read_csv(path)

df = df[desiredColumns]
print("partitions: %s" % df.npartitions)

df = df.head(1000000)


# https://pandas.pydata.org/pandas-docs/stable/api.html#groupby
group = df.groupby(by="OGLE3_ID")
for ogle3Id, index in group.groups.items():
    lc = df.iloc[index.values]
    # print("single group: %s" % lc)
    # print("ogle3 id: %s" % ogle3Id)
    cat = lc["OGLE3_CATEGORY"].iloc[0]
    times = np.array(lc["OGLE3_HJD"])
    mags = np.array(lc["OGLE3_MAGNITUDE"])
    errors = np.array(lc["OGLE3_ERROR"])
    # print(times)
    # print(mags)
    # print(errors)


# print(len(df))
# print(df.memory_usage().compute())
# print(len(df))  # 302,894,386
# print(df["OGLE3_ID"].nunique().compute())  # 786623
