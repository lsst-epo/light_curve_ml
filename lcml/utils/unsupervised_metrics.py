from collections import namedtuple

from sklearn import metrics


EXTERNAL_METRICS = sorted(["adjustedMutualInformation", "adjustedRandScore",
                           "completeness", "fowlkesMallows", "homogeneity",
                           "vMeasure"])


ExternalClusterMetrics = namedtuple("ExternalClusterMetrics", EXTERNAL_METRICS)


INTERNAL_METRICS = sorted(["calinskiHarabaz", "silhouetteCoefficient"])


InternalClusterMetrics = namedtuple("InternalClusterMetrics", INTERNAL_METRICS)


def computeExternalMetrics(labels, predLabels) -> ExternalClusterMetrics:
    """External metrics evaluate clustering performance against labeled
    data."""
    ami = metrics.adjusted_mutual_info_score(labels, predLabels)
    ars = metrics.adjusted_rand_score(labels, predLabels)
    fm = metrics.fowlkes_mallows_score(labels, predLabels)
    h, c, v = metrics.homogeneity_completeness_v_measure(labels, predLabels)
    return ExternalClusterMetrics(ami, ars, c, fm, h, v)


def computeInternalMetrics(X, predLabels) -> InternalClusterMetrics:
    """Internal metrics evalute clustering performance using only the
    cluster assignments."""
    ch = metrics.calinski_harabaz_score(X, predLabels)
    sc = metrics.silhouette_score(X, predLabels)
    return InternalClusterMetrics(ch, sc)

