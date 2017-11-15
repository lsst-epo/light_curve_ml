
from __future__ import print_function

import numpy as np
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood
from nupic.algorithms.sdr_classifier import SDRClassifier
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from htmresearch.algorithms.union_temporal_pooler import UnionTemporalPooler


def sdrClassifierExample():
    # http://nupic.docs.numenta.org/stable/api/algorithms/classifiers.html
    """steps - Sequence of the different steps of multi-step predictions
    to learn
    alpha - learning rate (larger -> faster learning)
    actValueAlpha - Used to track the actual value within each bucket.
    A lower actValueAlpha results in longer term memory"""
    c = SDRClassifier(steps=[1], alpha=0.1, actValueAlpha=0.1, verbosity=0)

    # learning
    c.compute(recordNum=0, patternNZ=[1, 5, 9],
              classification={"bucketIdx": 4, "actValue": 34.7},
              learn=True, infer=False)

    # inference
    result = c.compute(recordNum=1, patternNZ=[1, 5, 9],
                       classification={"bucketIdx": 4, "actValue": 34.7},
                       learn=False, infer=True)

    # Print the top three predictions for 1 steps out.
    topPredictions = sorted(zip(result[1],
                                result["actualValues"]), reverse=True)[:3]
    for prob, value in topPredictions:
        print("Prediction of {} has prob: {}.".format(value, prob * 100.0))


def main():
    # cluster similar inputs together in SDR space
    s = SpatialPooler()
    print(type(s))

    # powerful sequence memory in SDR space
    t = TemporalMemory()
    print(type(t))

    # computes rolling Gaussian based on raw anomaly scores and then their
    # likelihood
    a = AnomalyLikelihood()
    print(type(a))

    # temporally groups active cell sets from TM
    u = UnionTemporalPooler()
    print(type(u))

    # learning pairings of Union representations and labeled classes
    c = SDRClassifier()
    print(type(c))


if __name__ == "__main__":
    main()
