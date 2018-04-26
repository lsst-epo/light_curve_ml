# N.B. sklearn KMeans consumes all memory and crashes python
from collections import namedtuple
from datetime import timedelta
import time
from typing import List

from prettytable import PrettyTable
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.externals.joblib import Memory

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.format_util import truncatedFloat
from lcml.utils.unsupervised_metrics import (EXTERNAL_METRICS, INTERNAL_METRICS,
                                             computeExternalMetrics,
                                             computeInternalMetrics)


logger = BasicLogging.getLogger(__name__)


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self, features, labels, classToLabel):
        clusters = self.searchParams["clusterValues"]
        miniKMeansKwargs = self.searchParams["miniBatchKMeansArgs"]
        aggKwargs = self.searchParams["agglomerativeArgs"]
        aggKwargs["memory"] = Memory(cachedir=aggKwargs["memory"])

        kMeansName = "mini-batch k-means"
        agglomNameToLinkage = {"agg-ward": "ward", "agg-complete": "complete",
                               "agg-average": "average"}
        scores = {k: list() for k in list(agglomNameToLinkage) + [kMeansName]}
        for c in clusters:
            logger.info("clusters: %s", c)
            model = MiniBatchKMeans(n_clusters=c, **miniKMeansKwargs)
            self.evaluateClusteringModel(model, labels, features,
                                         scores[kMeansName], kMeansName)
            del model

            # for now we try all linkages
            for aggName, linkage in agglomNameToLinkage.items():
                affinity = "euclidean" if linkage == "ward" else "manhattan"
                model = AgglomerativeClustering(n_clusters=c, linkage=linkage,
                                                affinity=affinity, **aggKwargs)
                self.evaluateClusteringModel(model, labels, features,
                                             scores[aggName], aggName)
                del model

        clusterTable = PrettyTable(["clusters", "algorithm"] +
                                   sorted(EXTERNAL_METRICS) +
                                   sorted(INTERNAL_METRICS))
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
