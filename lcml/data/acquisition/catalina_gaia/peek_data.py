#!/usr/bin/env python
"""Parsing and viewing various datasets."""
import csv
import os

import numpy as np

from lcml.utils import dataset_util, format_util
import lcml.utils.pathing


def loadDataset(dataName, datasetName, useDeltaEncoder=False):
  fileDir = os.path.join('./{}'.format(datasetName),
                         dataName, dataName+'_TRAIN')
  trainData = np.loadtxt(fileDir, delimiter=',')
  trainLabel = trainData[:, 0].astype('int')
  trainData = trainData[:, 1:]

  fileDir = os.path.join('./{}'.format(datasetName),
                         dataName, dataName + '_TEST')
  testData = np.loadtxt(fileDir, delimiter=',')
  testLabel = testData[:, 0].astype('int')
  testData = testData[:, 1:]

  if useDeltaEncoder:
    trainData = np.diff(trainData)
    testData = np.diff(testData)

  classList = np.unique(trainLabel)
  classMap = {}
  for i in range(len(classList)):
    classMap[classList[i]] = i

  for i in range(len(trainLabel)):
    trainLabel[i] = classMap[trainLabel[i]]
  for i in range(len(testLabel)):
    testLabel[i] = classMap[testLabel[i]]

  return trainData, trainLabel, testData, testLabel


def _parseLightCurveCatalina(paths):
    """Parses a light curve time series from Catalina periodic dataset."""
    lcs = []
    labels = []
    for path in paths:
        with open(path, "r") as f:
            reader = csv.reader(f)

            # indices for Catalina data: 9 - MJD, 2 - magnitude
            lcs.append([(format_util.toDatetime(row[9]), row[2])
                        for i, row in enumerate(reader) if i])

            fileName = path.split("/")[-1]
            category = "_".join(fileName.split("_")[:-1]).lower()
            labels.append(category)

    return lcs, labels


def peekCatalina():
    datasetName = "catalina/periodic"
    extension = ".csv"
    paths = lcml.utils.pathing.getDatasetFilePaths(datasetName, extension)
    lightCurves, labels = _parseLightCurveCatalina(paths)
    dataset_util.reportDataset(lightCurves, labels)

    datasets = lightCurves[:1]
    for i, d in enumerate(datasets):
        print("\ncategory: %s" % labels[i])
        print("datetime\tmagnitude")
        for t, v in d:
            print("%s\t%s" % (t, v))


def peekGaia(sampleSize=11):
    paths = lcml.utils.pathing.getDatasetFilePaths("gaia", ".csv")
    for p in paths:
        if "16" not in p:
            continue

        print("\n" + p)
        data = np.genfromtxt(p, delimiter=",", dtype=None)
        print(data.shape)
        # column 3 - 'ref_epoch' - Reference epoch to which the astrometic
        # source parameters are referred, expressed as a Julian Year in TCB.
        #
        # column 49 - 'phot_g_mean_flux' - Mean flux in the G-band.

        # column 51 - 'phot_g_mean_mag' - Mean magnitude in the G band. This is
        # computed from the G-band mean flux applying the magnitude zero-point
        # in the Vega scale
        refEpoch = []
        for row in data[:sampleSize]:
            print("\t".join([row[3], row[49], row[51]]))

            try:
                refEpoch.append(float(row[3]))
            except ValueError:
                print(row[3])

        print("min: %s max: %s ave: %s" % (min(refEpoch), max(refEpoch),
                                           np.average(refEpoch)))


if __name__ == "__main__":
    peekGaia()
