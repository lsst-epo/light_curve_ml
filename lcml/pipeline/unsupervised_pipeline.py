from sklearn.cluster import KMeans

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.database.sqlite_db import connFromParams, selectFeatures
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self):
        # Method to load features

        conn = connFromParams(self.dbParams)
        cursor = conn.cursor()

        # TODO make this labels and features
        features = selectFeatures(cursor, self.dbParams)
        print(len(features))

        centroidStart = 4
        centroidEnd = 12
        for clusters in range(centroidStart, centroidEnd):
            model = KMeans(n_clusters=clusters)
            model.fit(features)
            print(len(model.labels_))
