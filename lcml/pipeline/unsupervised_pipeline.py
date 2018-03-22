from collections import namedtuple
from datetime import timedelta
import time

from prettytable import PrettyTable

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.database.sqlite_db import (connFromParams,
                                              selectLabelsFeatures)
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

    def modelSelectionPhase(self):
        conn = connFromParams(self.dbParams)
        cursor = conn.cursor()

        labels, features = selectLabelsFeatures(cursor, self.dbParams)
        logger.info("Loaded: %d features", len(features))

        dataLimit = self.selectionParams["dataLimit"]
        if dataLimit:
            labels = labels[:dataLimit]
            features = features[:dataLimit]
        clusterValues = self.selectionParams["clusterValues"]

        # n_init - number of runs starting from random initialization, bumping
        # may incr perf by avoiding local minima
        kMeansKwargs = self.selectionParams["miniBatchKMeansArgs"]
        aggKwargs = self.selectionParams["agglomerativeArgs"]

        db = DBSCAN(n_jobs=-1)
        dbStart = time.time()
        db.fit(features)
        logger.info("dbscan fit in: %s",
                    timedelta(seconds=(time.time() - dbStart)))
        kmScores = []
        agScores = []
        kmName = "mini-batch k-means"
        agName = "agglomerative"
        for c in clusterValues:
            logger.info("clusters: %s", c)
            kMeans = MiniBatchKMeans(n_clusters=c, **kMeansKwargs)
            self.evaluateClusteringModel(kMeans, kmName, labels, features,
                                         kmScores)

            agg = AgglomerativeClustering(n_clusters=c, **aggKwargs)
            self.evaluateClusteringModel(agg, agName, labels, features,
                                         agScores)

        externMetrics = sorted(vars(kmScores[0][0]))
        internMetrics = sorted(vars(kmScores[0][1]))
        clusterTable = PrettyTable(["clusters", "clustering algorithm"] +
                                   externMetrics + internMetrics)

        nameScores = [(kmName, kmScores), (agName, agScores)]
        for i, c in enumerate(clusterValues):
            for name, scores in nameScores:
                row = ([c, name] + self._scoreValues(scores[i][0]) +
                       self._scoreValues(scores[i][1]))
                clusterTable.add_row(row)
        logger.info("\n" + str(clusterTable))

    @staticmethod
    def _scoreValues(scores):
        """Converts nametuple to a list of values in key-sorted order"""
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
