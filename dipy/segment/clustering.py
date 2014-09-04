from dipy.segment.clusteringspeed import quickbundles
from dipy.segment.metric import MDF


class Clustering:
    def cluster(data):
        raise NotImplementedError("Subclass has to define this function!")


class QuickBundles(Clustering):
    def __init__(self, threshold, metric=MDF()):
        self.threshold = threshold
        self.metric = metric

    def cluster(self, data):
        return quickbundles(data, self.metric, threshold=self.threshold)
