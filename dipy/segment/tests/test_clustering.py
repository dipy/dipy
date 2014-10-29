
import numpy as np
import itertools

from dipy.segment.clusteringspeed import Cluster, ClusterCentroid
from dipy.segment.clusteringspeed import ClusterMap, ClusterMapCentroid

from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_raises, run_module_suite
from dipy.testing import assert_arrays_equal


features_shape = (1, 10)
dtype = "float32"
features = np.ones(features_shape, dtype=dtype)

data = [np.arange(3*5, dtype=dtype).reshape((-1, 3)),
        np.arange(3*10, dtype=dtype).reshape((-1, 3)),
        np.arange(3*15, dtype=dtype).reshape((-1, 3)),
        np.arange(3*17, dtype=dtype).reshape((-1, 3)),
        np.arange(3*20, dtype=dtype).reshape((-1, 3))]

expected_clusters = [[2, 4], [0, 3], [1]]


def test_cluster_attributes_and_constructor():
    clusters = ClusterMap()
    cluster = clusters.add_cluster()
    assert_equal(type(cluster), Cluster)

    assert_equal(cluster.id, 0)
    assert_raises(AttributeError, setattr, cluster, 'id', 0)
    assert_raises(AttributeError, setattr, cluster, 'indices', [])
    assert_array_equal(cluster.indices, [])
    assert_equal(len(cluster), 0)

    # Cannot create cluster that unexists in cluster map.
    assert_raises(ValueError, Cluster, None, 0)
    assert_raises(ValueError, Cluster, clusters, -1)
    assert_raises(ValueError, Cluster, clusters, 42)

    # Duplicate
    assert_equal(cluster, Cluster(clusters, cluster.id))


def test_cluster_add():
    clusters = ClusterMap()
    cluster = clusters.add_cluster()

    indices = []
    for idx in range(1, 10):
        cluster.add(idx)
        indices.append(idx)
        assert_equal(len(cluster), idx)
        assert_array_equal(cluster.indices, indices)

    # Test add multiples indices at the same time
    cluster = clusters.add_cluster()
    cluster.add(*range(1, 10))
    assert_array_equal(cluster.indices, indices)


def test_cluster_iter():
    indices = range(len(data))
    np.random.shuffle(indices)  # None trivial ordering

    # Test without specifying refdata in ClusterMap
    clusters = ClusterMap()
    cluster = clusters.add_cluster()
    cluster.add(*indices)
    assert_array_equal(cluster.indices, indices)
    assert_array_equal(list(cluster), indices)

    # Test with specifying refdata in ClusterMap
    clusters.refdata = data
    assert_arrays_equal(list(cluster), [data[i] for i in indices])


def test_cluster_getitem():
    indices = range(len(data))
    np.random.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    # Test without specifying refdata in ClusterMap
    clusters = ClusterMap()
    cluster = clusters.add_cluster()
    cluster.add(*indices)

    # Test indexing
    for i in advanced_indices:
        assert_equal(cluster[i], indices[i])

    # Test advanced indexing
    assert_array_equal(cluster[advanced_indices], [indices[i] for i in advanced_indices])

    # Test index out of bound
    assert_raises(IndexError, cluster.__getitem__, len(cluster))
    assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

    # Test slicing and negative indexing
    assert_equal(cluster[-1], indices[-1])
    assert_array_equal(cluster[::2], indices[::2])
    assert_arrays_equal(cluster[::-1], indices[::-1])
    assert_arrays_equal(cluster[:-1], indices[:-1])
    assert_arrays_equal(cluster[1:], indices[1:])

    # Test with specifying refdata in ClusterMap
    clusters.refdata = data

    # Test indexing
    for i in advanced_indices:
        assert_array_equal(cluster[i], data[indices[i]])

    # Test advanced indexing
    assert_array_equal(cluster[advanced_indices], [data[indices[i]] for i in advanced_indices])

    # Test index out of bound
    assert_raises(IndexError, cluster.__getitem__, len(cluster))
    assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

    # Test slicing and negative indexing
    assert_array_equal(cluster[-1], data[indices[-1]])
    assert_arrays_equal(cluster[::2], [data[i] for i in indices[::2]])
    assert_arrays_equal(cluster[::-1], [data[i] for i in indices[::-1]])
    assert_arrays_equal(cluster[:-1], [data[i] for i in indices[:-1]])
    assert_arrays_equal(cluster[1:], [data[i] for i in indices[1:]])


def test_cluster_centroid_attributes_and_constructor():
    clusters = ClusterMapCentroid(features_shape)
    cluster = clusters.add_cluster()
    assert_equal(type(cluster), ClusterCentroid)

    assert_equal(cluster.id, 0)
    assert_raises(AttributeError, setattr, cluster, 'id', 0)
    assert_raises(AttributeError, setattr, cluster, 'indices', [])
    assert_raises(AttributeError, setattr, cluster, 'centroid', [])
    assert_array_equal(cluster.indices, [])
    assert_array_equal(cluster.centroid, np.zeros(features_shape))
    assert_equal(len(cluster), 0)

    # Cannot create cluster that unexists in cluster map.
    assert_raises(ValueError, ClusterCentroid, None, 0)
    assert_raises(ValueError, ClusterCentroid, clusters, -1)
    assert_raises(ValueError, ClusterCentroid, clusters, 42)

    # Duplicate
    assert_equal(cluster, ClusterCentroid(clusters, cluster.id))


def test_cluster_centroid_add():
    clusters = ClusterMapCentroid(features_shape)
    cluster = clusters.add_cluster()

    indices = []
    centroid = np.zeros(features_shape, dtype=dtype)
    for idx in range(1, 10):
        cluster.add(idx, (idx+1)*features)
        indices.append(idx)
        centroid = (centroid * (idx-1) + (idx+1)*features) / idx
        assert_equal(len(cluster), idx)
        assert_array_equal(cluster.indices, indices)
        assert_array_equal(cluster.centroid, centroid)


def test_cluster_centroid_iter():
    indices = range(len(data))
    np.random.shuffle(indices)  # None trivial ordering

    # Test without specifying refdata in ClusterMapCentroid
    clusters = ClusterMapCentroid(features_shape)
    cluster = clusters.add_cluster()
    for idx in indices:
        cluster.add(idx, (idx+1)*features)

    assert_array_equal(cluster.indices, indices)
    assert_array_equal(list(cluster), indices)

    # Test with specifying refdata in ClusterMapCentroid
    clusters.refdata = data
    assert_arrays_equal(list(cluster), [data[i] for i in indices])


def test_cluster_centroid_getitem():
    indices = range(len(data))
    np.random.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    # Test without specifying refdata in ClusterMapCentroid
    clusters = ClusterMapCentroid(features_shape)
    cluster = clusters.add_cluster()
    for idx in indices:
        cluster.add(idx, (idx+1)*features)

    # Test indexing
    for i in advanced_indices:
        assert_equal(cluster[i], indices[i])

    # Test advanced indexing
    assert_array_equal(cluster[advanced_indices], [indices[i] for i in advanced_indices])

    # Test index out of bound
    assert_raises(IndexError, cluster.__getitem__, len(cluster))
    assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

    # Test slicing and negative indexing
    assert_equal(cluster[-1], indices[-1])
    assert_array_equal(cluster[::2], indices[::2])
    assert_arrays_equal(cluster[::-1], indices[::-1])
    assert_arrays_equal(cluster[:-1], indices[:-1])
    assert_arrays_equal(cluster[1:], indices[1:])

    # Test with specifying refdata in ClusterMapCentroid
    clusters.refdata = data

    # Test indexing
    for i in advanced_indices:
        assert_array_equal(cluster[i], data[indices[i]])

    # Test advanced indexing
    assert_array_equal(cluster[advanced_indices], [data[indices[i]] for i in advanced_indices])

    # Test index out of bound
    assert_raises(IndexError, cluster.__getitem__, len(cluster))
    assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

    # Test slicing and negative indexing
    assert_array_equal(cluster[-1], data[indices[-1]])
    assert_arrays_equal(cluster[::2], [data[i] for i in indices[::2]])
    assert_arrays_equal(cluster[::-1], [data[i] for i in indices[::-1]])
    assert_arrays_equal(cluster[:-1], [data[i] for i in indices[:-1]])
    assert_arrays_equal(cluster[1:], [data[i] for i in indices[1:]])


def test_cluster_map_attributes_and_constructor():
    clusters = ClusterMap()
    assert_equal(len(clusters), 0)
    assert_array_equal(clusters.clusters, [])
    assert_array_equal(list(clusters), [])
    assert_raises(IndexError, clusters.__getitem__, 0)
    assert_raises(AttributeError, setattr, clusters, 'clusters', [])


def test_cluster_map_add_cluster():
    clusters = ClusterMap()

    list_of_indices = []
    for i in range(3):
        list_of_indices.append([])
        cluster = clusters.add_cluster()
        assert_equal(type(cluster), Cluster)
        assert_equal(len(clusters), i+1)
        assert_equal(cluster, clusters[cluster.id])

        for id_data in range(2*i):
            list_of_indices[-1].append(id_data)
            cluster.add(id_data)

    assert_array_equal(list(itertools.chain(*clusters)), list(itertools.chain(*list_of_indices)))


def test_cluster_map_iter():
    nb_clusters = 11

    # Test without specifying refdata in ClusterMap
    cluster_map = ClusterMap()
    clusters = []
    for i in range(nb_clusters):
        clusters.append(cluster_map.add_cluster())

    assert_array_equal([cluster.id for cluster in cluster_map.clusters], range(nb_clusters))
    assert_array_equal([cluster.id for cluster in cluster_map], range(nb_clusters))
    assert_array_equal(cluster_map.clusters, clusters)
    assert_array_equal(cluster_map, clusters)


def test_cluster_map_getitem():
    nb_clusters = 11
    indices = range(len(data))
    np.random.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    # Test without specifying refdata in ClusterMap
    cluster_map = ClusterMap()
    clusters = []
    for i in range(nb_clusters):
        clusters.append(cluster_map.add_cluster())

    # Test indexing
    for i in advanced_indices:
        assert_equal(cluster_map[i], clusters[i])

    # Test advanced indexing
    assert_array_equal(cluster_map[advanced_indices], [clusters[i] for i in advanced_indices])

    # Test index out of bound
    assert_raises(IndexError, cluster_map.__getitem__, len(clusters))
    assert_raises(IndexError, cluster_map.__getitem__, -len(clusters)-1)

    # Test slicing and negative indexing
    assert_equal(cluster_map[-1], clusters[-1])
    assert_array_equal(cluster_map[::2], clusters[::2])
    assert_arrays_equal(cluster_map[::-1], clusters[::-1])
    assert_arrays_equal(cluster_map[:-1], clusters[:-1])
    assert_arrays_equal(cluster_map[1:], clusters[1:])


def test_cluster_map_comparison_with_int():
    clusters1_indices = range(10)
    clusters2_indices = range(10, 15)
    clusters3_indices = [15]

    # Build a test ClusterMap
    clusters = ClusterMap()
    cluster1 = clusters.add_cluster()
    cluster1.add(*clusters1_indices)
    cluster2 = clusters.add_cluster()
    cluster2.add(*clusters2_indices)
    cluster3 = clusters.add_cluster()
    cluster3.add(*clusters3_indices)

    subset = clusters < 5
    assert_equal(subset.sum(), 1)
    assert_array_equal(list(clusters[subset][0]), clusters3_indices)

    subset = clusters <= 5
    assert_equal(subset.sum(), 2)
    assert_array_equal(list(clusters[subset][0]), clusters2_indices)
    assert_array_equal(list(clusters[subset][1]), clusters3_indices)

    subset = clusters == 5
    assert_equal(subset.sum(), 1)
    assert_array_equal(list(clusters[subset][0]), clusters2_indices)

    subset = clusters != 5
    assert_equal(subset.sum(), 2)
    assert_array_equal(list(clusters[subset][0]), clusters1_indices)
    assert_array_equal(list(clusters[subset][1]), clusters3_indices)

    subset = clusters > 5
    assert_equal(subset.sum(), 1)
    assert_array_equal(list(clusters[subset][0]), clusters1_indices)

    subset = clusters >= 5
    assert_equal(subset.sum(), 2)
    assert_array_equal(list(clusters[subset][0]), clusters1_indices)
    assert_array_equal(list(clusters[subset][1]), clusters2_indices)


def test_cluster_map_centroid_attributes_and_constructor():
    clusters = ClusterMapCentroid(features_shape)
    assert_array_equal(clusters.centroids, [])
    assert_raises(AttributeError, setattr, clusters, 'centroids', [])


def test_cluster_map_centroid_add_cluster():
    clusters = ClusterMapCentroid(features_shape)

    centroids = []
    for i in range(3):
        cluster = clusters.add_cluster()
        assert_array_equal(cluster.centroid, np.zeros_like(features))
        assert_equal(type(cluster), ClusterCentroid)
        assert_equal(cluster, clusters[cluster.id])

        centroids.append(np.zeros_like(features))
        for id_data in range(2*i):
            centroids[-1] = (centroids[-1]*id_data + (id_data+1)*features) / (id_data+1)
            cluster.add(id_data, (id_data+1)*features)

    assert_array_equal(list(itertools.chain(*clusters.centroids)), list(itertools.chain(*centroids)))

    # Check adding features of different sizes (shorter and longer)
    features_shape_short = (1, features_shape[1]-3)
    features_too_short = np.ones(features_shape_short, dtype=dtype)
    cluster = clusters.add_cluster()
    assert_raises(ValueError, cluster.add, 123, features_too_short)

    features_shape_long = (1, features_shape[1]+3)
    features_too_long = np.ones(features_shape_long, dtype=dtype)
    cluster = clusters.add_cluster()
    assert_raises(ValueError, cluster.add, 123, features_too_long)


def test_cluster_map_centroid_iter():
    nb_clusters = 11

    # Test without specifying refdata in ClusterMapCentroid
    cluster_map = ClusterMapCentroid(features_shape)
    clusters = []
    for i in range(nb_clusters):
        clusters.append(cluster_map.add_cluster())

    assert_array_equal([cluster.id for cluster in cluster_map.clusters], range(nb_clusters))
    assert_array_equal([cluster.id for cluster in cluster_map], range(nb_clusters))
    assert_array_equal(cluster_map.clusters, clusters)
    assert_array_equal(cluster_map, clusters)


def test_cluster_map_centroid_getitem():
    nb_clusters = 11
    indices = range(len(data))
    np.random.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    # Test without specifying refdata in ClusterMapCentroid
    cluster_map = ClusterMapCentroid(features_shape)
    clusters = []
    for i in range(nb_clusters):
        clusters.append(cluster_map.add_cluster())

    # Test indexing
    for i in advanced_indices:
        assert_equal(cluster_map[i], clusters[i])

    # Test advanced indexing
    assert_array_equal(cluster_map[advanced_indices], [clusters[i] for i in advanced_indices])

    # Test index out of bound
    assert_raises(IndexError, cluster_map.__getitem__, len(clusters))
    assert_raises(IndexError, cluster_map.__getitem__, -len(clusters)-1)

    # Test slicing and negative indexing
    assert_equal(cluster_map[-1], clusters[-1])
    assert_array_equal(cluster_map[::2], clusters[::2])
    assert_arrays_equal(cluster_map[::-1], clusters[::-1])
    assert_arrays_equal(cluster_map[:-1], clusters[:-1])
    assert_arrays_equal(cluster_map[1:], clusters[1:])


def test_cluster_map_centroid_comparison_with_int():
    clusters1_indices = range(10)
    clusters2_indices = range(10, 15)
    clusters3_indices = [15]

    # Build a test ClusterMapCentroid
    clusters = ClusterMapCentroid(features_shape)
    cluster1 = clusters.add_cluster()
    for i in clusters1_indices:
        cluster1.add(i, features)

    cluster2 = clusters.add_cluster()
    for i in clusters2_indices:
        cluster2.add(i, features)

    cluster3 = clusters.add_cluster()
    for i in clusters3_indices:
        cluster3.add(i, features)

    subset = clusters < 5
    assert_equal(subset.sum(), 1)
    assert_array_equal(list(clusters[subset][0]), clusters3_indices)

    subset = clusters <= 5
    assert_equal(subset.sum(), 2)
    assert_array_equal(list(clusters[subset][0]), clusters2_indices)
    assert_array_equal(list(clusters[subset][1]), clusters3_indices)

    subset = clusters == 5
    assert_equal(subset.sum(), 1)
    assert_array_equal(list(clusters[subset][0]), clusters2_indices)

    subset = clusters != 5
    assert_equal(subset.sum(), 2)
    assert_array_equal(list(clusters[subset][0]), clusters1_indices)
    assert_array_equal(list(clusters[subset][1]), clusters3_indices)

    subset = clusters > 5
    assert_equal(subset.sum(), 1)
    assert_array_equal(list(clusters[subset][0]), clusters1_indices)

    subset = clusters >= 5
    assert_equal(subset.sum(), 2)
    assert_array_equal(list(clusters[subset][0]), clusters1_indices)
    assert_array_equal(list(clusters[subset][1]), clusters2_indices)


if __name__ == '__main__':
    run_module_suite()
