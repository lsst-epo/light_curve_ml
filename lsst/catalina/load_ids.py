from pandas import DataFrame, read_csv

import matplotlib.pyplot as plt
import pandas as pd
import sys
import matplotlib

path = "../../data/catalina/gcvs/Catalina_GCVS.vars"


df = pd.read_csv(path, delim_whitespace=True)
print df.dtypes
s = set(df["Type"])

print s
