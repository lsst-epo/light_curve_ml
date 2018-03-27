from collections import namedtuple
from datetime import timedelta
import time

from joblib import Memory
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


ExternalClusterMetrics = namedtuple("ExternalClusterMetrics",
                                    ["adjustedRandScore",
                                     "adjustedMutualInformation",
                                     "homogeneity", "completeness",
                                     "vMeasure", "fowlkesMallows"])


InternalClusterMetrics = namedtuple("InternalClusterMetrics",
                                    ["silhouetteCoefficient",
                                     "calinskiHarabaz"])


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self, trainFeatures, trainLabels, classToLabel):
        clusterValues = self.searchParams["clusterValues"]
        miniKMeansKwargs = self.searchParams["miniBatchKMeansArgs"]
        kmeansKwargs = self.searchParams["kmeansArgs"]
        aggKwargs = self.searchParams["agglomerativeArgs"]

        miniKmName = "mini-batch k-means"
        # kmName = "k-means"
        aggNameToLinkage = {"agg-ward": "ward", "agg-complete": "complete",
                            "agg-average": "average"}
        nameToScores = {k: list()
                        for k in list(aggNameToLinkage) + [miniKmName]}
        aggKwargs["memory"] = Memory(cachedir=aggKwargs["memory"])
        for c in clusterValues:
            logger.info("clusters: %s", c)
            miniKMeans = MiniBatchKMeans(n_clusters=c, **miniKMeansKwargs)
            self.evaluateClusteringModel(miniKMeans, miniKmName, trainLabels,
                                         trainFeatures, nameToScores[miniKmName]
                                         )
            del miniKMeans


            # N.B. consumes all memory and crashes python
            # kMeans = KMeans(n_clusters=c, **kmeansKwargs)
            # self.evaluateClusteringModel(kMeans, kmName, labels, features,
            #                              nameToScores[kmName])
            # del kMeans

            # for now we try all linkages
            for aggName, linkage in aggNameToLinkage.items():
                affinity = "euclidean" if linkage == "ward" else "manhattan"
                agg = AgglomerativeClustering(n_clusters=c, linkage=linkage,
                                              affinity=affinity, **aggKwargs)
                self.evaluateClusteringModel(agg, aggName, trainLabels,
                                             trainFeatures,
                                             nameToScores[aggName])
                del agg

        externMetrics = sorted(vars(nameToScores[miniKmName][0][0]))
        internMetrics = sorted(vars(nameToScores[miniKmName][0][1]))
        clusterTable = PrettyTable(["clusters", "clustering algorithm"] +
                                   externMetrics + internMetrics)
        for i, c in enumerate(clusterValues):
            for name, scores in nameToScores.items():
                row = ([c, name] + self._scoreValues(scores[i][0]) +
                       self._scoreValues(scores[i][1]))
                clusterTable.add_row(row)
        logger.info("\n" + str(clusterTable))

    @staticmethod
    def _scoreValues(scores):
        """Converts namedtuple to a list of values in key-sorted order"""
        return ["%.4f" % kv[1] for kv in sorted(vars(scores).items())]

    @staticmethod
    def evaluateClusteringModel(model, modelName, labels, features, scores):
        s = time.time()
        model.fit(features)
        logger.info("%s fit in: %s", modelName,
                    timedelta(seconds=(time.time() - s)))
        external = UnsupervisedPipeline.computeExternalMetrics(labels,
                                                               model.labels_)
        internal = UnsupervisedPipeline.computeInternalMetrics(features,
                                                               model.labels_)
        scores.append((external, internal))
        return scores

    @staticmethod
    def computeExternalMetrics(labels, predLabels):
        """External metrics evaluate clustering performance against labeled
        data."""
        rs = metrics.adjusted_rand_score(labels, predLabels)
        ami = metrics.adjusted_mutual_info_score(labels, predLabels)
        h, c, v = metrics.homogeneity_completeness_v_measure(labels, predLabels)
        fmi = metrics.fowlkes_mallows_score(labels, predLabels)
        return ExternalClusterMetrics(rs, ami, h, c, v, fmi)

    @staticmethod
    def computeInternalMetrics(X, predLabels):
        """Internal metrics evalute clustering performance using only the
        cluster assignments."""
        sc = metrics.silhouette_score(X, predLabels)
        ch = metrics.calinski_harabaz_score(X, predLabels)
        return InternalClusterMetrics(sc, ch)

    def evaluateTestSet(self, model, featuresTest, labelsTest, classLabels):
        # FIXME
        pass
