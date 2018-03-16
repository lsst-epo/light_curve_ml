from sklearn.cluster import KMeans

from lcml.pipeline.batch_pipeline import BatchPipeline
from lcml.utils.basic_logging import BasicLogging


logger = BasicLogging.getLogger(__name__)


class UnsupervisedPipeline(BatchPipeline):
    def __init__(self, conf):
        BatchPipeline.__init__(self, conf)

    def modelSelectionPhase(self):
        # look into a good abstraction for the
        centroidStart = 4
        centroidEnd = 12
        for clusters in range(centroidStart, centroidEnd):

            model = KMeans(n_clusters=clusters)

            # xIris has dimensionality: [n_samples, n_features]
            # model.fit(xIris)
            # clustering = model.labels_[::10]
            # print(clustering)
            # print(yIris[::10])
