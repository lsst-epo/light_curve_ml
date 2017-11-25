#!/usr/bin/env python
import os

import numpy as np
from prettytable import PrettyTable
import upsilon


_GARBAGE_VALUES = {float("NaN"), float("-Inf"), float("Inf")}


def _toFloat(v):
    f = float(v)
    if f in _GARBAGE_VALUES:
        raise ValueError("Garbage value")

    return f


def confidenceInterval(data, numStds):
    mean = np.mean(data)
    std = np.std(data)
    return mean - numStds * std, mean + numStds * std


def removeOutliers(values, errors, numStds=3):
    """Given a time series of values and their associated errors returns a copy
    of both arrays with the outliers removed. If either the value or the error
    extends beyond the specified number of standard deviations, the entire
    point (value and error) is dropped."""
    valLwr, valUpr = confidenceInterval(values, numStds)
    errLwr, errUpr = confidenceInterval(errors, numStds)

    return zip(*[(v, errors[i]) for i, v in enumerate(values)
                 if valLwr < v < valUpr and errLwr < errors[i] < errUpr])


def removeMachoOutliers(mjds, values, errors, threshold=-99.0):
    return zip(*[(mjds[i], v, errors[i]) for i, v in enumerate(values)
                 if v > threshold and errors[i] > threshold])


SUFFICIENT_DATA = 80


def loadMacho(maxRows=10000):
    lightCurves = []
    dataDir = os.path.join(os.environ.get("LSST"), "data/macho")
    for fieldDir in os.listdir(dataDir):
        fieldPath = os.path.join(dataDir, fieldDir)
        if not os.path.isdir(fieldPath):
            continue

        for file in os.listdir(fieldPath):
            fileDir = os.path.join(fieldPath, file)
            trainData = np.genfromtxt(fileDir, delimiter=';', dtype=None,
                                      max_rows=maxRows)
            if len(trainData) < SUFFICIENT_DATA:
                continue

            mjds = []
            redMags = []
            redErrors = []
            blueMags = []
            blueErrors = []
            for line in trainData:
                try:
                    mjd = _toFloat(line[4])
                    rMag = _toFloat(line[9])
                    rErr = _toFloat(line[10])
                    bMag = _toFloat(line[24])
                    bErr = _toFloat(line[25])
                except ValueError:
                    print("bad value: %s" % line)
                    continue

                mjds.append(mjd)
                redMags.append(rMag)
                redErrors.append(rErr)
                blueMags.append(bMag)
                blueErrors.append(bErr)

            redMjds, clRedMags, clRedErrors = removeMachoOutliers(mjds, redMags,
                                                                  redErrors)
            blueMjds, clBlueMags, clBlueErrors = removeMachoOutliers(mjds,
                                                                     blueMags,
                                                                     blueErrors)
            t = PrettyTable(field_names=["type", "original", "cleaned",
                                         "removed"])
            t.add_row(["red", len(redMags), len(clRedMags),
                       len(redMags) - len(clRedMags)])
            t.add_row(["blue", len(blueMags), len(clBlueMags),
                       len(blueMags) - len(clBlueMags)])
            print(t)
            lightCurves.append([redMjds, clRedMags, clRedErrors])
            lightCurves.append([blueMjds, clBlueMags, clBlueErrors])

    return lightCurves


def main():
    print("Loading data...")
    set_of_light_curves = loadMacho()

    # Loads baked classification model
    print("\nLoading classifier...")
    rf_model = upsilon.load_rf_model()

    for datetimes, magnitudes, errors in set_of_light_curves:
        # Extract features from each light curve and predict its class.
        date = np.array(datetimes)
        mag = np.array(magnitudes)
        err = np.array(errors)

        try:
            # Extract features
            print("\nExtracting features...")
            e_features = upsilon.ExtractFeatures(date, mag, err, n_threads=3)
            e_features.run()
            features = e_features.get_features()

            # Classify the light curve
            print("Classifying...")
            label, probability, flag = upsilon.predict(rf_model, features)
            print([label, probability, flag])
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()
