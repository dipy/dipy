import operator
import numpy as np

from abc import ABCMeta, abstractmethod

from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric


class Identity:
    """ Provides identity indexing functionality.

    This can replace any class supporting indexing used for referencing
    (e.g. list, tuple). Indexing an instance of this class will return the
    index provided instead of the element. It does not support slicing.
    """
    def __getitem__(self, idx):
        return idx


class Cluster(object):
    """ Provides functionalities for interacting with a cluster.

    Useful container to retrieve index of elements grouped together. If
    a reference to the data is provided to `cluster_map`, elements will
    be returned instead of their index when possible.

    Parameters
    ----------
    cluster_map : `ClusterMap` object
        Reference to the set of clusters this cluster is being part of.
    id : int
        Id of this cluster in its associated `cluster_map` object.
    refdata : list (optional)
        Actual elements that clustered indices refer to.

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieve them using its `ClusterMap` object.
    """
    def __init__(self, id=0, indices=None, refdata=Identity()):
        self.id = id
        self.refdata = refdata
        self.indices = indices if indices is not None else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """ Gets element(s) through indexing.

        If a reference to the data was provided (via refdata property)
        elements will be returned instead of their index.

        Parameters
        ----------
        idx : int, slice or list
            Index of the element(s) to get.

        Returns
        -------
        `Cluster` object(s)
            When `idx` is a int, returns a single element.

            When `idx` is either a slice or a list, returns a list of elements.
        """
        if isinstance(idx, int) or isinstance(idx, np.integer):
            return self.refdata[self.indices[idx]]
        elif type(idx) is slice:
            return [self.refdata[i] for i in self.indices[idx]]
        elif type(idx) is list:
            return [self[i] for i in idx]

        msg = "Index must be a int or a slice! Not '{0}'".format(type(idx))
        raise TypeError(msg)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __str__(self):
        return "[" + ", ".join(map(str, self.indices)) + "]"

    def __repr__(self):
        return "Cluster(" + str(self) + ")"

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.indices == other.indices

    def __ne__(self, other):
        return not self == other

    def __cmp__(self, other):
        raise TypeError("Cannot compare Cluster objects.")

    def assign(self, *indices):
        """ Assigns indices to this cluster.

        Parameters
        ----------
        *indices : list of indices
            Indices to add to this cluster.
        """
        self.indices += indices


class ClusterCentroid(Cluster):
    """ Provides functionalities for interacting with a cluster.

    Useful container to retrieve the indices of elements grouped together and
    the cluster's centroid. If a reference to the data is provided to
    `cluster_map`, elements will be returned instead of their index when
    possible.

    Parameters
    ----------
    cluster_map : `ClusterMapCentroid` object
        Reference to the set of clusters this cluster is being part of.
    id : int
        Id of this cluster in its associated `cluster_map` object.
    refdata : list (optional)
        Actual elements that clustered indices refer to.

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieve them using its `ClusterMapCentroid` object.
    """

    def __init__(self, centroid, id=0, indices=None, refdata=Identity()):
        super(ClusterCentroid, self).__init__(id, indices, refdata)
        self.centroid = centroid.copy()
        self.new_centroid = centroid.copy()

    def __eq__(self, other):
        return (isinstance(other, ClusterCentroid) and
                np.all(self.centroid == other.centroid) and
                super(ClusterCentroid, self).__eq__(other))

    def assign(self, id_datum, features):
        """ Assigns a data point to this cluster.

        Parameters
        ----------
        id_datum : int
            Index of the data point to add to this cluster.
        features : 2D array
            Data point's features to modify this cluster's centroid.
        """
        N = len(self)
        self.new_centroid = ((self.new_centroid * N) + features) / (N+1.)
        super(ClusterCentroid, self).assign(id_datum)

    def update(self):
        """ Update centroid of this cluster.

        Returns
        -------
        converged : bool
            Tells if the centroid has moved.
        """
        converged = np.equal(self.centroid, self.new_centroid)
        self.centroid = self.new_centroid.copy()
        return converged


class ClusterMap(object):
    """ Provides functionalities for interacting with clustering outputs.

    Useful container to create, remove, retrieve and filter clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `Cluster` objects.

    Parameters
    ----------
    refdata : list
        Actual elements that clustered indices refer to.
    """
    def __init__(self, refdata=Identity()):
        self._clusters = []
        self.refdata = refdata

    @property
    def clusters(self):
        return self._clusters

    @property
    def refdata(self):
        return self._refdata

    @refdata.setter
    def refdata(self, value):
        if value is None:
            value = Identity()

        self._refdata = value
        for cluster in self.clusters:
            cluster.refdata = self._refdata

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        """ Gets cluster(s) through indexing.

        Parameters
        ----------
        idx : int, slice, list or boolean array
            Index of the element(s) to get.

        Returns
        -------
        `Cluster` object(s)
            When `idx` is a int, returns a single `Cluster` object.

            When `idx`is either a slice, list or boolean array, returns
            a list of `Cluster` objects.
        """
        if isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            return [self.clusters[i]
                    for i, take_it in enumerate(idx) if take_it]
        elif type(idx) is slice:
            return [self.clusters[i] for i in range(*idx.indices(len(self)))]
        elif type(idx) is list:
            return [self.clusters[i] for i in idx]

        return self.clusters[idx]

    def __iter__(self):
        return iter(self.clusters)

    def __str__(self):
        return "[" + ", ".join(map(str, self)) + "]"

    def __repr__(self):
        return "ClusterMap(" + str(self) + ")"

    def _richcmp(self, other, op):
        """ Compares this cluster map with another cluster map or an integer.

        Two `ClusterMap` objects are equal if they contain the same clusters.
        When comparing a `ClusterMap` object with an integer, the comparison
        will be performed on the size of the clusters instead.

        Parameters
        ----------
        other : `ClusterMap` object or int
            Object to compare to.
        op : rich comparison operators (see module `operator`)
            Valid operators are: lt, le, eq, ne, gt or ge.

        Returns
        -------
        bool or 1D array (bool)
            When comparing to another `ClusterMap` object, it returns whether
            the two `ClusterMap` objects contain the same clusters or not.

            When comparing to an integer the comparison is performed on the
            clusters sizes, it returns an array of boolean.
        """
        if isinstance(other, ClusterMap):
            if op is operator.eq:
                return isinstance(other, ClusterMap) \
                    and len(self) == len(other) \
                    and self.clusters == other.clusters
            elif op is operator.ne:
                return not self == other

            raise NotImplementedError(
                "Can only check if two ClusterMap instances are equal or not.")

        elif isinstance(other, int):
            return np.array([op(len(cluster), other) for cluster in self])

        msg = ("ClusterMap only supports comparison with a int or another"
               " instance of Clustermap.")
        raise NotImplementedError(msg)

    def __eq__(self, other):
        return self._richcmp(other, operator.eq)

    def __ne__(self, other):
        return self._richcmp(other, operator.ne)

    def __lt__(self, other):
        return self._richcmp(other, operator.lt)

    def __le__(self, other):
        return self._richcmp(other, operator.le)

    def __gt__(self, other):
        return self._richcmp(other, operator.gt)

    def __ge__(self, other):
        return self._richcmp(other, operator.ge)

    def add_cluster(self, *clusters):
        """ Adds one or multiple clusters to this cluster map.

        Parameters
        ----------
        *clusters : `Cluster` object, ...
            Cluster(s) to be added in this cluster map.
        """
        for cluster in clusters:
            self.clusters.append(cluster)
            cluster.refdata = self.refdata

    def remove_cluster(self, *clusters):
        """ Remove one or multiple clusters from this cluster map.

        Parameters
        ----------
        *clusters : `Cluster` object, ...
            Cluster(s) to be removed from this cluster map.
        """
        for cluster in clusters:
            self.clusters.remove(cluster)

    def clear(self):
        """ Remove all clusters from this cluster map. """
        del self.clusters[:]

    def size(self):
        """ Gets number of clusters contained in this cluster map. """
        return len(self)

    def clusters_sizes(self):
        """ Gets the size of every cluster contained in this cluster map.

        Returns
        -------
        list of int
            Sizes of every cluster in this cluster map.
        """
        return list(map(len, self))

    def get_large_clusters(self, min_size):
        """ Gets clusters which contains at least `min_size` elements.

        Parameters
        ----------
        min_size : int
            Minimum number of elements a cluster needs to have to be selected.

        Returns
        -------
        list of `Cluster` objects
            Clusters having at least `min_size` elements.
        """
        return self[self >= min_size]

    def get_small_clusters(self, max_size):
        """ Gets clusters which contains at most `max_size` elements.

        Parameters
        ----------
        max_size : int
            Maximum number of elements a cluster can have to be selected.

        Returns
        -------
        list of `Cluster` objects
            Clusters having at most `max_size` elements.
        """
        return self[self <= max_size]


class ClusterMapCentroid(ClusterMap):
    """ Provides functionalities for interacting with clustering outputs
    that have centroids.

    Allows to retrieve easely the centroid of every cluster. Also, it is
    a useful container to create, remove, retrieve and filter clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `ClusterCentroid` objects.

    Parameters
    ----------
    refdata : list
        Actual elements that clustered indices refer to.
    """
    @property
    def centroids(self):
        return [cluster.centroid for cluster in self.clusters]


class Clustering(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def cluster(self, data, ordering=None):
        """ Clusters `data`.

        Subclasses will perform their clustering algorithm here.

        Parameters
        ----------
        data : list of N-dimensional arrays
            Each array represents a data point.
        ordering : iterable of indices, optional
            Specifies the order in which data points will be clustered.

        Returns
        -------
        `ClusterMap` object
            Result of the clustering.
        """
        msg = "Subclass has to define method 'cluster(data, ordering)'!"
        raise NotImplementedError(msg)


class QuickBundles(Clustering):
    r""" Clusters streamlines using QuickBundles [Garyfallidis12]_.

    Given a list of streamlines, the QuickBundles algorithm sequentially
    assigns each streamline to its closest bundle in $\mathcal{O}(Nk)$ where
    $N$ is the number of streamlines and $k$ is the final number of bundles.
    If for a given streamline its closest bundle is farther than `threshold`,
    a new bundle is created and the streamline is assigned to it except if the
    number of bundles has already exceeded `max_nb_clusters`.

    Parameters
    ----------
    threshold : float
        The maximum distance from a bundle for a streamline to be still
        considered as part of it.
    metric : str or `Metric` object (optional)
        The distance metric to use when comparing two streamlines. By default,
        the Minimum average Direct-Flip (MDF) distance [Garyfallidis12]_ is
        used and streamlines are automatically resampled so they have
        12 points.
    max_nb_clusters : int
        Limits the creation of bundles.

    Examples
    --------
    >>> from dipy.segment.clustering import QuickBundles
    >>> from dipy.data import get_data
    >>> from nibabel import trackvis as tv
    >>> streams, hdr = tv.read(get_data('fornix'))
    >>> streamlines = [i[0] for i in streams]
    >>> # Segment fornix with a treshold of 10mm and streamlines resampled
    >>> # to 12 points.
    >>> qb = QuickBundles(threshold=10.)
    >>> clusters = qb.cluster(streamlines)
    >>> len(clusters)
    4
    >>> list(map(len, clusters))
    [61, 191, 47, 1]
    >>> # Resampling streamlines differently is done explicitly as follows.
    >>> # Note this has an impact on the speed and the accuracy (tradeoff).
    >>> from dipy.segment.metric import ResampleFeature
    >>> from dipy.segment.metric import AveragePointwiseEuclideanMetric
    >>> feature = ResampleFeature(nb_points=2)
    >>> metric = AveragePointwiseEuclideanMetric(feature)
    >>> qb = QuickBundles(threshold=10., metric=metric)
    >>> clusters = qb.cluster(streamlines)
    >>> len(clusters)
    4
    >>> list(map(len, clusters))
    [58, 142, 72, 28]


    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """

    def __init__(self, threshold, metric="MDF_12points",
                 max_nb_clusters=np.iinfo('i4').max):
        self.threshold = threshold
        self.max_nb_clusters = max_nb_clusters

        if isinstance(metric, Metric):
            self.metric = metric
        elif metric == "MDF_12points":
            feature = ResampleFeature(nb_points=12)
            self.metric = AveragePointwiseEuclideanMetric(feature)
        else:
            raise ValueError("Unknown metric: {0}".format(metric))

    def cluster(self, streamlines, ordering=None):
        """ Clusters `streamlines` into bundles.

        Performs quickbundles algorithm using predefined metric and threshold.

        Parameters
        ----------
        streamlines : list of 2D arrays
            Each 2D array represents a sequence of 3D points (points, 3).
        ordering : iterable of indices
            Specifies the order in which data points will be clustered.

        Returns
        -------
        `ClusterMapCentroid` object
            Result of the clustering.
        """
        from dipy.segment.clustering_algorithms import quickbundles
        cluster_map = quickbundles(streamlines, self.metric,
                                   threshold=self.threshold,
                                   max_nb_clusters=self.max_nb_clusters,
                                   ordering=ordering)

        cluster_map.refdata = streamlines
        return cluster_map
