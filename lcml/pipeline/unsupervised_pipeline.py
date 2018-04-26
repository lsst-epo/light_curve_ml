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
AGGLOM_NAME_TO_LINKAGE = {"agg-ward": "ward",
                          "agg-complete": "complete",
                          "agg-average": "average"}


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)
        aggKwargs = self.searchParams["agglomerativeArgs"]
        aggKwargs["memory"] = Memory(cachedir=aggKwargs["memory"])

    def modelSelectionPhase(self, X: List[np.ndarray], y: List[str],
                            classToLabel: Dict[int, str]):
        logger.info("feature vectors: %s", len(X))
        X_normed = StandardScaler().fit_transform(X)
        pcaVarianceExplained = []
        ldaVarianceExplained = []
        components = list(range(self.searchParams["componentsStart"],
                                self.searchParams["componentsStop"],
                                self.searchParams["componentsStep"]))
        for c in components:
            pca = PCA(n_components=c)
            s = time.time()
            pcaReduced = pca.fit_transform(X_normed)
            logger.info("pca in %s", time.time() - s)
            pcaVarExp = sum(pca.explained_variance_ratio_)
            pcaVarianceExplained.append(pcaVarExp)

            lda = LinearDiscriminantAnalysis(n_components=c)
            s = time.time()
            ldaReduced = lda.fit_transform(X_normed, y)
            logger.info("lda in %s", time.time() - s)
            ldaVarExp = sum(lda.explained_variance_ratio_)
            ldaVarianceExplained.append(ldaVarExp)
            logger.info("components: %s PCA: %s LDA: %s", c, pcaVarExp,
                        ldaVarExp)
            logger.info("model selection for pca reduced...")
            self._modelSelection(pcaReduced, y)

            logger.info("selection for lda reduced...")
            self._modelSelection(ldaReduced, y)

    def _modelSelection(self, features: List[np.ndarray], labels: List[str]):
        # TODO pass in the type of reduction, the number of components, and the
        # TODO variance explained. include these in the table as first column

        # TODO also pass in the table from caller and only print out once
        clusters = self.searchParams["clusterValues"]
        kMeansKwargs = self.searchParams["miniBatchKMeansArgs"]
        aggKwargs = self.searchParams["agglomerativeArgs"]

        scores = {k: list()
                  for k in list(AGGLOM_NAME_TO_LINKAGE) + [KMEANS_NAME]}
        for c in clusters:
            logger.info("clusters: %s", c)
            model = MiniBatchKMeans(n_clusters=c, **kMeansKwargs)
            self.evaluateClusteringModel(model, labels, features,
                                         scores[KMEANS_NAME], KMEANS_NAME)
            del model

            # for now we try all linkages
            for aggName, linkage in AGGLOM_NAME_TO_LINKAGE.items():
                affinity = "euclidean" if linkage == "ward" else "manhattan"
                model = AgglomerativeClustering(n_clusters=c, linkage=linkage,
                                                affinity=affinity, **aggKwargs)
                self.evaluateClusteringModel(model, labels, features,
                                             scores[aggName], aggName)
                del model

        clusterTable = PrettyTable(["clusters", "algorithm"] + EXTERNAL_METRICS
                                   + INTERNAL_METRICS)
        for i, c in enumerate(clusters):
            for name, intExtScores in scores.items():
                row = ([c, name] + self._metricsToList(intExtScores[i][0]) +
                       self._metricsToList(intExtScores[i][1]))
                clusterTable.add_row(row)
        logger.info("\n" + str(clusterTable))

    @staticmethod
    def _metricsToList(scores: namedtuple, places: int=4) -> List[str]:
        """Converts namedtuple of scores to a list of rounded values in name
        sorted order"""
        tf = truncatedFloat(places)
        return [tf % v for _, v in sorted(scores._asdict().items())]

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
