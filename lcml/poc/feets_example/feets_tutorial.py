#!/usr/bin/env python3
"""Try out Feets tutorial"""
import time

from feets import datasets, FeatureSpace, preprocess
from matplotlib import pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plotLc2(lc):
    f = plt.figure(1)
    plt.plot(lc.bands.B.time, lc.bands.B.magnitude, "*-", alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.gca().invert_yaxis()
    f.show()


def plotLc(lc):
    f = plt.figure(1)
    plt.plot(lc.bands.B.time, lc.bands.B.magnitude, "*-", alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.gca().invert_yaxis()
    f.show()


def tutorial():
    plot = 0
    lc = datasets.load_MACHO_example()
    if plot:
        plotLc(lc)

    # 69 features with ALL_DATA_TYPES in 6.15s
    # 64 features with time, magnitude, error in 6.14s
    # 58 features with time, magnitude in 3.3s
    # 22 features with magnitude in 0.02s
    basicData = ["time", "magnitude", "error"]
    basicData = ["time", "magnitude"]
    basicData = ["magnitude"]

    start = time.time()

    # remove points beyond 5 stds of the mean
    tm, mag, error = preprocess.remove_noise(**lc.bands.B)
    tm2, mag2, error2 = preprocess.remove_noise(**lc.bands.R)

    # N.B. skip for LSST work
    aTime, aMag, aMag2, aError, aError2 = preprocess.align(tm, tm2, mag,
                                                           mag2, error, error2)
    lc = [tm, mag, error, mag2, aTime, aMag, aMag2, aError, aError2]

    # only calculate these features
    # fs = feets.FeatureSpace(only=['Std', 'StetsonL'])

    fs = FeatureSpace()
    # fs = FeatureSpace(data=basicData)
    features, values = fs.extract(*lc)

    elapsed = time.time() - start
    print("Computed %s features in %.02fs" % (len(features), elapsed))

    if plot:
        g = plt.figure(2)
        plt.plot(lc[0], lc[1], "*-", alpha=0.6)
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        g.show()
        input()

    t = PrettyTable(["Feature", "Value"])
    t.align = "l"
    for i, feat in enumerate(features):
        t.add_row([feat, values[i]])

    if plot:
        print(t)

    fdict = dict(zip(features, values))

    # Ploting the example lightcurve in phase
    T = 2 * fdict["PeriodLS"]
    new_b = np.mod(lc[0], T) / T
    idx = np.argsort(2 * new_b)

    plt.plot(new_b, lc[1], '*')
    plt.xlabel("Phase")
    plt.ylabel("Magnitude")
    plt.gca().invert_yaxis()
    plt.show()
