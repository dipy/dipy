from dipy.segment.clusteringspeed import quickbundles
from dipy.segment.metric import Metric
from dipy.segment.metric import AveragePointwiseEuclideanMetric


class Clustering:
    def cluster(data):
        raise NotImplementedError("Subclass has to define this function!")


class QuickBundles(Clustering):
    def __init__(self, threshold, metric="mdf"):
        self.threshold = threshold

        if isinstance(metric, Metric):
            self.metric = metric
        else:  # Assume metric contain the name of the metric to use:
            self.metric = AveragePointwiseEuclideanMetric()

    def cluster(self, data):
        return quickbundles(data, self.metric, threshold=self.threshold)
