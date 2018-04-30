# N.B. sklearn KMeans consumes all memory and crashes python
from collections import namedtuple
from datetime import timedelta
import time
from typing import Dict, List

import numpy as np
from prettytable import PrettyTable
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals.joblib import Memory
from sklearn.preprocessing import StandardScaler

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.format_util import truncatedFloat
from lcml.utils.unsupervised_metrics import (EXTERNAL_METRICS, INTERNAL_METRICS,
                                             computeExternalMetrics,
                                             computeInternalMetrics)


logger = BasicLogging.getLogger(__name__)


KMEANS_NAME = "mini-batch k-means"
DEFAULT_LINKAGES = {"agglomerative (ward)": "ward",
                    "agglomerative (complete)": "complete",
                    "agglomerative (average)": "average"}
_INITIAL_TABLE_COLUMNS = ["reduction algorithm", "components",
                          "variance explained", "clusters", "cluster algorithm"]


reduceStage = namedtuple("ReduceStage", ["components", "modelName",
                                         "modelClass"])


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)
        aggKwargs = self.searchParams["agglomerativeArgs"]
        aggKwargs["memory"] = Memory(cachedir=aggKwargs["memory"])
        self.linkages = aggKwargs.pop("linkages", DEFAULT_LINKAGES)
        self.places = self.globalParams["places"]

    def modelSelectionPhase(self, X: List[np.ndarray], y: List[str],
                            classToLabel: Dict[int, str]):
        logger.info("feature vectors: %s", len(X))
        X_normed = StandardScaler().fit_transform(X)
        components = list(range(self.searchParams["componentsStart"],
                                self.searchParams["componentsStop"],
                                self.searchParams["componentsStep"]))

        allRows = list()
        tests = [[reduceStage(c, "pca", PCA)] for c in components]
        tests += [[reduceStage(c, "lda", LinearDiscriminantAnalysis)]
                   for c in components]
        tests += [[reduceStage(c1, "pca", PCA),
                   reduceStage(c2, "lda", LinearDiscriminantAnalysis)]
                   for c1 in components for c2 in components]
        tests += [[reduceStage(c2, "lda", LinearDiscriminantAnalysis),
                   reduceStage(c1, "pca", PCA)]
                   for c1 in components for c2 in components]
        for i, test in enumerate(tests):
            logger.info("running test: %s", i)
            self._runDimReduct(X_normed, y, test, allRows)

        self._reportAllResults(allRows)
        self._reportBestMetrics(allRows)

    def _runDimReduct(self, X, y, test: List[namedtuple], allRows: list):
        XReduced = X
        varianceExplained = ""
        modelName = ""
        components = ""
        for stage in test:
            model = stage.modelClass(n_components=stage.components)
            XReduced = model.fit_transform(XReduced, y)
            varianceExplained += str(round(sum(model.explained_variance_ratio_),
                                     self.places)) + " "
            modelName += stage.modelName + " "
            components += str(stage.components) + " "

        rows = self._runClusters(XReduced, y)
        for r in rows:
            allRows.append([modelName, components, varianceExplained] + r)

    def _runClusters(self, features: List[np.ndarray],
                     labels: List[str]) -> List[List[str]]:
        clusters = self.searchParams["clusterValues"]
        kMeansKwargs = self.searchParams["miniBatchKMeansArgs"]
        aggKwargs = self.searchParams["agglomerativeArgs"]

        allScores = {k: list() for k in list(self.linkages) + [KMEANS_NAME]}
        for c in clusters:
            logger.info("clusters: %s", c)
            model = MiniBatchKMeans(n_clusters=c, **kMeansKwargs)
            self.evaluateClusteringModel(model, labels, features,
                                         allScores[KMEANS_NAME], KMEANS_NAME)
            del model

            for aggName, linkage in self.linkages.items():
                affinity = "euclidean" if linkage == "ward" else "manhattan"
                model = AgglomerativeClustering(n_clusters=c, linkage=linkage,
                                                affinity=affinity, **aggKwargs)
                self.evaluateClusteringModel(model, labels, features,
                                             allScores[aggName], aggName)
                del model

        rows = []
        for i, c in enumerate(clusters):
            for name, scores in allScores.items():
                rows.append([c, name] +
                            self._asSortedList(scores[i][0], self.places) +
                            self._asSortedList(scores[i][1], self.places))
        return rows

    @staticmethod
    def _asSortedList(scores: namedtuple, places: int=4) -> List[str]:
        """Converts namedtuple of scores to a list of rounded values in name
        sorted order"""
        tf = truncatedFloat(places)
        return [tf % v for _, v in sorted(scores._asdict().items())]


    @staticmethod
    def _reportAllResults(allRows):
        t = PrettyTable(_INITIAL_TABLE_COLUMNS + EXTERNAL_METRICS +
                        INTERNAL_METRICS)
        for r in allRows:
            t.add_row(r)
        logger.info("\n" + str(t))

    def _reportBestMetrics(self, allRows):
        t = PrettyTable(["metric", "max value"] + _INITIAL_TABLE_COLUMNS)
        for name, index in [("adjMutualInfo", 5), ("adjustedRand", 6),
                            ("fowlkesMallows", 8), ("calinskiHarabaz", 11),
                            ("silhouetteCoef", 12)]:
            t.add_row(self._addMaxAndConditions(allRows, name, index))
        logger.info("\n" + str(t))

    @staticmethod
    def _addMaxAndConditions(rows: List[List[str]], metricName: str,
                             index: int) -> list:
        maxInd = None
        maxVal = -float("inf")
        for i, r in enumerate(rows):
            if float(r[index]) > maxVal:
                maxInd = index
                maxVal = float(r[index])

        return [metricName, maxVal] + rows[maxInd][:5]

    @staticmethod
    def evaluateClusteringModel(model, labels, features, scores, name=""):
        s = time.time()
        model.fit(features)
        logger.info("%s fit in: %s", name,
                    timedelta(seconds=(time.time() - s)))
        external = computeExternalMetrics(labels, model.labels_)
        internal = computeInternalMetrics(features, model.labels_)
        scores.append((external, internal))
        return scores

    def evaluateTestSet(self, model, featuresTest, labelsTest, classLabels):
        pass
