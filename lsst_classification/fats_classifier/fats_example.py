#!/usr/bin/env python
""" To calculate all of the features requires:
magnitude, time, error, magnitude2, time2, error2"""
from __future__ import print_function

import numpy as np
from FATS import FeatureSpace


#: example magnitude data
MAG_EX = [0.46057565, 0.5137294, 0.70136533, 0.21454228,
          0.547923, 0.33433717, 0.4487987, 0.55571062,
          0.24388037, 0.44793366, 0.30175873, 0.88326381,
          0.12208977, 0.37088649, 0.5945731, 0.74705894,
          0.24551664, 0.36009236, 0.80661981, 0.04961063,
          0.87747311, 0.97388975, 0.95775496, 0.34195989,
          0.54201036, 0.87854618, 0.07388174, 0.21543205,
          0.59295337, 0.56771493]


#: example time data
TIME_EX = [float(n) for n in range(len(MAG_EX))]


def main():
    lc = np.array([MAG_EX, TIME_EX])

    # specify a list of features as input by specifying the features as a list
    # for the parameter featureList
    a = FeatureSpace(featureList=['Std'])
    a = a.calculateFeature(lc)
    result = a.result(method="dict")
    print(result)


    #
    # a = FeatureSpace(Data=["magnitude", "time"])
    # a = a.calculateFeature(lc)




if __name__ == "__main__":
    main()
