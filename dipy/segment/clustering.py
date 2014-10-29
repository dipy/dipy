import numpy as np
from dipy.segment.clustering_algorithms import quickbundles
from dipy.segment.metric import Metric
from dipy.segment.metric import AveragePointwiseEuclideanMetric


class Clustering:
    def cluster(self, data, ordering=None):
        """ Clusters `data`.

        Subclasses will perform their clustering algorithm here.

        Parameters
        ----------
        data : list of N-dimensional array
            each array represents a data point.
        ordering : iterable of indices
            change `data` ordering when applying the clustering algorithm

        Returns
        -------
        clusters : `ClusterMap` object
            result of the clustering
        """
        raise NotImplementedError("Subclass has to define this function!")


class QuickBundles(Clustering):
    r""" Clusters streamlines using QuickBundles algorithm.

    Given a list of streamlines, the QuickBundles algorithm sequentially
    assigns each streamline to its closest bundle in $\mathcal{O}(Nk)$ where
    $N$ is the number of streamlines and $k$ is the final number of bundles.
    If for a given streamline its closest bundle is farther than `threshold`,
    a new bundle is created and the streamline is assigned to it except if the
    number of bundles has already exceeded `max_nb_clusters`.

    Parameters
    ----------
    threshold : float
        criterion to consider a streamline as being part of a bundle
    metric : str or `Metric` object
        will be used to compute distance between two streamlines
    max_nb_clusters : int
        limits the creation of the bundles

    Metric names
    ------------
    *mdf* : Minimum Average Direct-Flip

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """
    def __init__(self, threshold, metric="mdf", max_nb_clusters=np.iinfo('i4').max):
        self.threshold = threshold
        self.max_nb_clusters = max_nb_clusters

        if isinstance(metric, Metric):
            self.metric = metric
        elif metric.lower() == "mdf":
            self.metric = AveragePointwiseEuclideanMetric()

    def cluster(self, streamlines, ordering=None):
        """ Clusters `streamlines` into bundles.

        Performs quickbundles algorithm using predefined metric and threshold.

        Parameters
        ----------
        streamlines : list of 2D array
            each 2D array represents a sequence of 3D points (points, 3).
        ordering : iterable of indices
            change `streamlines` ordering when applying QuickBundles

        Returns
        -------
        clusters : `ClusterMapCentroid` object
            result of the clustering
        """
        return quickbundles(streamlines, self.metric,
                            threshold=self.threshold,
                            max_nb_clusters=self.max_nb_clusters,
                            ordering=ordering)
