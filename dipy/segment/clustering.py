
from dipy.segment.clusteringspeed import CentroidClusters


class Clustering:
    def cluster(data):
        raise NotImplementedError("Subclass has to define this function!")


class ClusterMap:
    def __init__(self, clusters, data=None, metric=None):
        self.data = data
        self.clusters = clusters
        self.metric = metric

    def centroids(self):
        return [self.metric.centroid(cluster) for cluster in self.clusters]

    def medoids(self):
        return [self.metric.medoid(cluster) for cluster in self.clusters]

    def __getitem__(self, slice):
        if self.data is None:
            return self.clusters[slice]

        return map(self.data.__getitem__, self.clusters[slice])
        #return [self.data[id] for id in self.clusters[slice]]

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        if self.data is None:
            return iter(self.clusters)

        return (map(self.data.__getitem__, cluster) for cluster in self.clusters)
        #return ([self.data[id] for id in cluster] for cluster in self.clusters)
