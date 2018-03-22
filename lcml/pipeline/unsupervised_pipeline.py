from sklearn.cluster import AgglomerativeClustering, KMeans

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.database.sqlite_db import (connFromParams,
                                              selectLabelsFeatures)
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.dataset_util import convertClassLabels


logger = BasicLogging.getLogger(__name__)


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self):
        conn = connFromParams(self.dbParams)
        cursor = conn.cursor()

        labels, features = selectLabelsFeatures(cursor, self.dbParams)
        intLabels, classToLabel = convertClassLabels(labels)
        logger.info("Loaded: %d features", len(features))
        clusterValues = self.selectionParams["clusterValues"]

        # n_init - number of runs starting from random initialization, bumping
        # may incr perf by avoiding local minima
        kMeansKwargs = self.selectionParams["kmeansArgs"]
        aggKwargs = self.selectionParams["agglomerativeArgs"]
        sample = 10
        for c in clusterValues:
            kMeans = KMeans(n_clusters=c, **kMeansKwargs)
            kMeans.fit(features)

            kMeansLabels = kMeans.labels_
            print(len(kMeans.labels_))

            ag = AgglomerativeClustering(n_clusters=c, **aggKwargs)
            ag.fit(features)

            print("\nclusters: %s", c)
            print("labels: %s", labels[:sample])
            print("kmeans: %s", kMeans.labels_[:sample])
            print("aggl: %s", ag.labels_[:sample])
