from sklearn.cluster import KMeans

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.pipeline.database.sqlite_db import connFromParams, selectLabelsAndFeatures
from lcml.utils.basic_logging import BasicLogging
from lcml.utils.dataset_util import convertClassLabels


logger = BasicLogging.getLogger(__name__)


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self):
        conn = connFromParams(self.dbParams)
        cursor = conn.cursor()

        labels, features = selectLabelsAndFeatures(cursor, self.dbParams)
        intLabels, classToLabel = convertClassLabels(labels)
        logger.info("Loaded features: %d", len(features))

        print(len(features))
        centroidStart = 4
        centroidEnd = 12

        # TODO bump to improve performance
        nInit = 10  # number of runs starting from random initialization
        maxIter = 300   # max iterations for a single run
        nJobs = -1
        precomputeDists = True
        algorithm = "elkan"
        copyX = False
        kwargs = {"n_init": nInit, "max_iter": maxIter, "n_jobs": nJobs,
                  "algorithm": algorithm, "copy_x": copyX,
                  "precompute_distances": precomputeDists}
        for clusters in range(centroidStart, centroidEnd):
            model = KMeans(n_clusters=clusters, **kwargs)
            model.fit(features)

            kMeansLabels = model.labels_
            print(len(model.labels_))
