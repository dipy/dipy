import copy
import itertools

import numpy as np
from numpy.testing import assert_array_equal, assert_raises, assert_equal

from dipy.segment.clustering import Cluster, ClusterCentroid
from dipy.segment.clustering import ClusterMap, ClusterMapCentroid
from dipy.segment.clustering import Clustering

from dipy.testing import assert_true, assert_false, assert_arrays_equal
from dipy.testing.decorators import set_random_number_generator


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
    cluster = Cluster()
    assert_equal(type(cluster), Cluster)

    assert_equal(cluster.id, 0)
    assert_array_equal(cluster.indices, [])
    assert_equal(len(cluster), 0)

    # Duplicate
    assert_true(cluster == Cluster(cluster.id,
                                   cluster.indices,
                                   cluster.refdata))
    assert_false(cluster != Cluster(cluster.id,
                                    cluster.indices,
                                    cluster.refdata))

    # Invalid comparison
    assert_raises(TypeError, cluster.__cmp__, cluster)


def test_cluster_assign():
    cluster = Cluster()

    indices = []
    for idx in range(1, 10):
        cluster.assign(idx)
        indices.append(idx)
        assert_equal(len(cluster), idx)
        assert_equal(type(cluster.indices), list)
        assert_array_equal(cluster.indices, indices)

    # Test add multiples indices at the same time
    cluster = Cluster()
    cluster.assign(*range(1, 10))
    assert_array_equal(cluster.indices, indices)


@set_random_number_generator()
def test_cluster_iter(rng):
    indices = list(range(len(data)))
    rng.shuffle(indices)  # None trivial ordering

    # Test without specifying refdata
    cluster = Cluster()
    cluster.assign(*indices)
    assert_array_equal(cluster.indices, indices)
    assert_array_equal(list(cluster), indices)

    # Test with specifying refdata in ClusterMap
    cluster.refdata = data
    assert_arrays_equal(list(cluster), [data[i] for i in indices])


@set_random_number_generator()
def test_cluster_getitem(rng):
    indices = list(range(len(data)))
    rng.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    # Test without specifying refdata in ClusterMap
    cluster = Cluster()
    cluster.assign(*indices)

    # Test indexing
    for i in advanced_indices:
        assert_equal(cluster[i], indices[i])

    # Test advanced indexing
    assert_array_equal(cluster[advanced_indices],
                       [indices[i] for i in advanced_indices])

    # Test index out of bounds
    assert_raises(IndexError, cluster.__getitem__, len(cluster))
    assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

    # Test slicing and negative indexing
    assert_equal(cluster[-1], indices[-1])
    assert_array_equal(cluster[::2], indices[::2])
    assert_arrays_equal(cluster[::-1], indices[::-1])
    assert_arrays_equal(cluster[:-1], indices[:-1])
    assert_arrays_equal(cluster[1:], indices[1:])

    # Test with wrong indexing object
    assert_raises(TypeError, cluster.__getitem__, "wrong")

    # Test with specifying refdata in ClusterMap
    cluster.refdata = data

    # Test indexing
    for i in advanced_indices:
        assert_array_equal(cluster[i], data[indices[i]])

    # Test advanced indexing
    assert_arrays_equal(cluster[advanced_indices],
                        [data[indices[i]] for i in advanced_indices])

    # Test index out of bounds
    assert_raises(IndexError, cluster.__getitem__, len(cluster))
    assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

    # Test slicing and negative indexing
    assert_array_equal(cluster[-1], data[indices[-1]])
    assert_arrays_equal(cluster[::2], [data[i] for i in indices[::2]])
    assert_arrays_equal(cluster[::-1], [data[i] for i in indices[::-1]])
    assert_arrays_equal(cluster[:-1], [data[i] for i in indices[:-1]])
    assert_arrays_equal(cluster[1:], [data[i] for i in indices[1:]])

    # Test with wrong indexing object
    assert_raises(TypeError, cluster.__getitem__, "wrong")


@set_random_number_generator()
def test_cluster_str_and_repr(rng):
    indices = list(range(len(data)))
    rng.shuffle(indices)  # None trivial ordering

    # Test without specifying refdata in ClusterMap
    cluster = Cluster()
    cluster.assign(*indices)
    assert_equal(str(cluster), "[" + ", ".join(map(str, indices)) + "]")
    assert_equal(repr(cluster),
                 "Cluster([" + ", ".join(map(str, indices)) + "])")

    # Test with specifying refdata in ClusterMap
    cluster.refdata = data
    assert_equal(str(cluster), "[" + ", ".join(map(str, indices)) + "]")
    assert_equal(repr(cluster),
                 "Cluster([" + ", ".join(map(str, indices)) + "])")


def test_cluster_centroid_attributes_and_constructor():
    centroid = np.zeros(features_shape)
    cluster = ClusterCentroid(centroid)
    assert_equal(type(cluster), ClusterCentroid)

    assert_equal(cluster.id, 0)
    assert_array_equal(cluster.indices, [])
    assert_array_equal(cluster.centroid, np.zeros(features_shape))
    assert_equal(len(cluster), 0)

    # Duplicate
    assert_true(cluster == ClusterCentroid(centroid))
    assert_false(cluster != ClusterCentroid(centroid))
    assert_false(cluster == ClusterCentroid(centroid+1))

    # Invalid comparison
    assert_raises(TypeError, cluster.__cmp__, cluster)


def test_cluster_centroid_assign():
    centroid = np.zeros(features_shape)
    cluster = ClusterCentroid(centroid)

    indices = []
    centroid = np.zeros(features_shape, dtype=dtype)
    for idx in range(1, 10):
        cluster.assign(idx, (idx+1) * features)
        cluster.update()
        indices.append(idx)
        centroid = (centroid * (idx-1) + (idx+1) * features) / idx
        assert_equal(len(cluster), idx)
        assert_equal(type(cluster.indices), list)
        assert_array_equal(cluster.indices, indices)
        assert_equal(type(cluster.centroid), np.ndarray)
        assert_array_equal(cluster.centroid, centroid)


@set_random_number_generator()
def test_cluster_centroid_iter(rng):
    indices = list(range(len(data)))
    rng.shuffle(indices)  # None trivial ordering

    # Test without specifying refdata in ClusterCentroid
    centroid = np.zeros(features_shape)
    cluster = ClusterCentroid(centroid)
    for idx in indices:
        cluster.assign(idx, (idx+1)*features)

    assert_array_equal(cluster.indices, indices)
    assert_array_equal(list(cluster), indices)

    # Test with specifying refdata in ClusterCentroid
    cluster.refdata = data
    assert_arrays_equal(list(cluster), [data[i] for i in indices])


@set_random_number_generator()
def test_cluster_centroid_getitem(rng):
    indices = list(range(len(data)))
    rng.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    # Test without specifying refdata in ClusterCentroid
    centroid = np.zeros(features_shape)
    cluster = ClusterCentroid(centroid)
    for idx in indices:
        cluster.assign(idx, (idx+1)*features)

    # Test indexing
    for i in advanced_indices:
        assert_equal(cluster[i], indices[i])

    # Test advanced indexing
    assert_array_equal(cluster[advanced_indices],
                       [indices[i] for i in advanced_indices])

    # Test index out of bounds
    assert_raises(IndexError, cluster.__getitem__, len(cluster))
    assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

    # Test slicing and negative indexing
    assert_equal(cluster[-1], indices[-1])
    assert_array_equal(cluster[::2], indices[::2])
    assert_arrays_equal(cluster[::-1], indices[::-1])
    assert_arrays_equal(cluster[:-1], indices[:-1])
    assert_arrays_equal(cluster[1:], indices[1:])

    # Test with specifying refdata in ClusterCentroid
    cluster.refdata = data

    # Test indexing
    for i in advanced_indices:
        assert_array_equal(cluster[i], data[indices[i]])

    # Test advanced indexing
    assert_arrays_equal(cluster[advanced_indices],
                        [data[indices[i]] for i in advanced_indices])

    # Test index out of bounds
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

    list_of_cluster_objects = []
    list_of_indices = []
    for i in range(3):
        cluster = Cluster()
        list_of_cluster_objects.append(cluster)
        list_of_indices.append([])

        for id_data in range(2 * i):
            list_of_indices[-1].append(id_data)
            cluster.assign(id_data)

        clusters.add_cluster(cluster)
        assert_equal(type(cluster), Cluster)
        assert_equal(len(clusters), i+1)
        assert_true(cluster == clusters[-1])

    assert_array_equal(list(itertools.chain(*clusters)),
                       list(itertools.chain(*list_of_indices)))

    # Test adding multiple clusters at once.
    clusters = ClusterMap()
    clusters.add_cluster(*list_of_cluster_objects)
    assert_array_equal(list(itertools.chain(*clusters)),
                       list(itertools.chain(*list_of_indices)))


def test_cluster_map_remove_cluster():
    clusters = ClusterMap()

    cluster1 = Cluster(indices=[1])
    clusters.add_cluster(cluster1)

    cluster2 = Cluster(indices=[1, 2])
    clusters.add_cluster(cluster2)

    cluster3 = Cluster(indices=[1, 2, 3])
    clusters.add_cluster(cluster3)

    assert_equal(len(clusters), 3)

    clusters.remove_cluster(cluster2)
    assert_equal(len(clusters), 2)
    assert_array_equal(list(itertools.chain(*clusters)),
                       list(itertools.chain(*[cluster1, cluster3])))
    assert_equal(clusters[0], cluster1)
    assert_equal(clusters[1], cluster3)

    clusters.remove_cluster(cluster3)
    assert_equal(len(clusters), 1)
    assert_array_equal(list(itertools.chain(*clusters)), list(cluster1))
    assert_equal(clusters[0], cluster1)

    clusters.remove_cluster(cluster1)
    assert_equal(len(clusters), 0)
    assert_array_equal(list(itertools.chain(*clusters)), [])

    # Test removing multiple clusters at once.
    clusters = ClusterMap()
    clusters.add_cluster(cluster1, cluster2, cluster3)

    clusters.remove_cluster(cluster3, cluster2)
    assert_equal(len(clusters), 1)
    assert_array_equal(list(itertools.chain(*clusters)), list(cluster1))
    assert_equal(clusters[0], cluster1)

    clusters = ClusterMap()
    clusters.add_cluster(cluster2, cluster1, cluster3)

    clusters.remove_cluster(cluster1, cluster3, cluster2)
    assert_equal(len(clusters), 0)
    assert_array_equal(list(itertools.chain(*clusters)), [])


def test_cluster_map_clear():
    nb_clusters = 11
    clusters = ClusterMap()
    for i in range(nb_clusters):
        new_cluster = Cluster(indices=range(i))
        clusters.add_cluster(new_cluster)

    clusters.clear()
    assert_equal(len(clusters), 0)
    assert_array_equal(list(itertools.chain(*clusters)), [])


@set_random_number_generator(42)
def test_cluster_map_iter(rng):
    nb_clusters = 11

    # Test without specifying refdata in ClusterMap
    cluster_map = ClusterMap()
    clusters = []
    for i in range(nb_clusters):
        new_cluster = Cluster(indices=rng.integers(0, len(data), size=10))
        cluster_map.add_cluster(new_cluster)
        clusters.append(new_cluster)

    assert_true(all([c1 is c2 for c1, c2 in zip(cluster_map.clusters,
                                                clusters)]))
    assert_array_equal(cluster_map, clusters)
    assert_array_equal(cluster_map.clusters, clusters)
    assert_array_equal(cluster_map, [cluster.indices for cluster in clusters])

    # Set refdata
    cluster_map.refdata = data
    for c1, c2 in zip(cluster_map, clusters):
        assert_arrays_equal(c1, [data[i] for i in c2.indices])

    # Remove refdata, i.e. back to indices
    cluster_map.refdata = None
    assert_array_equal(cluster_map, [cluster.indices for cluster in clusters])


@set_random_number_generator()
def test_cluster_map_getitem(rng):
    nb_clusters = 11
    indices = list(range(nb_clusters))
    rng.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    cluster_map = ClusterMap()
    clusters = []
    for i in range(nb_clusters):
        new_cluster = Cluster(indices=range(i))
        cluster_map.add_cluster(new_cluster)
        clusters.append(new_cluster)

    # Test indexing
    for i in advanced_indices:
        assert_true(cluster_map[i] == clusters[i])

    # Test advanced indexing
    assert_arrays_equal(cluster_map[advanced_indices],
                        [clusters[i] for i in advanced_indices])

    # Test index out of bounds
    assert_raises(IndexError, cluster_map.__getitem__, len(clusters))
    assert_raises(IndexError, cluster_map.__getitem__, -len(clusters)-1)

    # Test slicing and negative indexing
    assert_equal(cluster_map[-1], clusters[-1])
    assert_array_equal(np.array(cluster_map[::2], dtype=object),
                       np.array(clusters[::2], dtype=object))
    assert_arrays_equal(cluster_map[::-1], clusters[::-1])
    assert_arrays_equal(cluster_map[:-1], clusters[:-1])
    assert_arrays_equal(cluster_map[1:], clusters[1:])


def test_cluster_map_str_and_repr():
    nb_clusters = 11
    cluster_map = ClusterMap()
    clusters = []
    for i in range(nb_clusters):
        new_cluster = Cluster(indices=range(i))
        cluster_map.add_cluster(new_cluster)
        clusters.append(new_cluster)

    expected_str = "[" + ", ".join(map(str, clusters)) + "]"
    assert_equal(str(cluster_map), expected_str)
    assert_equal(repr(cluster_map), "ClusterMap(" + expected_str + ")")


def test_cluster_map_size():
    nb_clusters = 11
    cluster_map = ClusterMap()
    clusters = [Cluster() for _ in range(nb_clusters)]
    cluster_map.add_cluster(*clusters)

    assert_equal(len(cluster_map), nb_clusters)
    assert_equal(cluster_map.size(), nb_clusters)


@set_random_number_generator(42)
def test_cluster_map_clusters_sizes(rng):
    nb_clusters = 11
    # Generate random indices
    indices = [range(rng.integers(1, 10)) for _ in range(nb_clusters)]

    cluster_map = ClusterMap()
    clusters = [Cluster(indices=indices[i]) for i in range(nb_clusters)]
    cluster_map.add_cluster(*clusters)

    assert_equal(cluster_map.clusters_sizes(), list(map(len, indices)))


@set_random_number_generator(42)
def test_cluster_map_get_small_and_large_clusters(rng):
    nb_clusters = 11
    cluster_map = ClusterMap()

    # Randomly generate small clusters
    indices = [rng.integers(0, 10, size=i) for i in range(1, nb_clusters+1)]
    small_clusters = [Cluster(indices=indices[i]) for i in range(nb_clusters)]
    cluster_map.add_cluster(*small_clusters)

    # Randomly generate small clusters
    indices = [rng.integers(0, 10, size=i)
               for i in range(nb_clusters+1, 2*nb_clusters+1)]
    large_clusters = [Cluster(indices=indices[i]) for i in range(nb_clusters)]
    cluster_map.add_cluster(*large_clusters)

    assert_equal(len(cluster_map), 2*nb_clusters)
    assert_equal(len(cluster_map.get_small_clusters(nb_clusters)),
                 len(small_clusters))
    assert_arrays_equal(cluster_map.get_small_clusters(nb_clusters),
                        small_clusters)
    assert_equal(len(cluster_map.get_large_clusters(nb_clusters+1)),
                 len(large_clusters))
    assert_arrays_equal(cluster_map.get_large_clusters(nb_clusters+1),
                        large_clusters)


def test_cluster_map_comparison_with_int():
    clusters1_indices = range(10)
    clusters2_indices = range(10, 15)
    clusters3_indices = [15]

    # Build a test ClusterMap
    clusters = ClusterMap()
    cluster1 = Cluster()
    cluster1.assign(*clusters1_indices)
    clusters.add_cluster(cluster1)

    cluster2 = Cluster()
    cluster2.assign(*clusters2_indices)
    clusters.add_cluster(cluster2)

    cluster3 = Cluster()
    cluster3.assign(*clusters3_indices)
    clusters.add_cluster(cluster3)

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


def test_cluster_map_comparison_with_object():
    nb_clusters = 4
    cluster_map = ClusterMap()
    # clusters = []
    for i in range(nb_clusters):
        new_cluster = Cluster(indices=range(i))
        cluster_map.add_cluster(new_cluster)
        # clusters.append(new_cluster)

    # Comparison with another ClusterMap object
    other_cluster_map = copy.deepcopy(cluster_map)
    assert_equal(np.array(cluster_map, dtype=object),
                 np.array(other_cluster_map, dtype=object))

    other_cluster_map = copy.deepcopy(cluster_map)
    assert_false(cluster_map != other_cluster_map)

    other_cluster_map = copy.deepcopy(cluster_map)
    assert_raises(NotImplementedError, cluster_map.__le__, other_cluster_map)

    # Comparison with an object that is not a ClusterMap or int
    assert_raises(NotImplementedError, cluster_map.__le__, float(42))


def test_cluster_map_centroid_attributes_and_constructor():
    clusters = ClusterMapCentroid()
    assert_array_equal(clusters.centroids, [])
    assert_raises(AttributeError, setattr, clusters, 'centroids', [])


def test_cluster_map_centroid_add_cluster():
    clusters = ClusterMapCentroid()

    centroids = []
    for i in range(3):
        cluster = ClusterCentroid(centroid=np.zeros_like(features))

        centroids.append(np.zeros_like(features))
        for id_data in range(2*i):
            centroids[-1] = ((centroids[-1]*id_data + (id_data+1)*features) /
                             (id_data+1))
            cluster.assign(id_data, (id_data+1)*features)
            cluster.update()

        clusters.add_cluster(cluster)
        assert_array_equal(cluster.centroid, centroids[-1])
        assert_equal(type(cluster), ClusterCentroid)
        assert_true(cluster == clusters[-1])

    assert_equal(type(clusters.centroids), list)
    assert_array_equal(list(itertools.chain(*clusters.centroids)),
                       list(itertools.chain(*centroids)))

    # Check adding features of different sizes (shorter and longer)
    features_shape_short = (1, features_shape[1]-3)
    features_too_short = np.ones(features_shape_short, dtype=dtype)
    assert_raises(ValueError, cluster.assign, 123, features_too_short)

    features_shape_long = (1, features_shape[1]+3)
    features_too_long = np.ones(features_shape_long, dtype=dtype)
    assert_raises(ValueError, cluster.assign, 123, features_too_long)


@set_random_number_generator()
def test_cluster_map_centroid_remove_cluster(rng):
    clusters = ClusterMapCentroid()

    centroid1 = rng.random(features_shape).astype(dtype)
    cluster1 = ClusterCentroid(centroid1, indices=[1])
    clusters.add_cluster(cluster1)

    centroid2 = rng.random(features_shape).astype(dtype)
    cluster2 = ClusterCentroid(centroid2, indices=[1, 2])
    clusters.add_cluster(cluster2)

    centroid3 = rng.random(features_shape).astype(dtype)
    cluster3 = ClusterCentroid(centroid3, indices=[1, 2, 3])
    clusters.add_cluster(cluster3)

    assert_equal(len(clusters), 3)

    clusters.remove_cluster(cluster2)
    assert_equal(len(clusters), 2)
    assert_array_equal(list(itertools.chain(*clusters)),
                       list(itertools.chain(*[cluster1, cluster3])))
    assert_array_equal(clusters.centroids, np.array([centroid1, centroid3]))
    assert_equal(clusters[0], cluster1)
    assert_equal(clusters[1], cluster3)

    clusters.remove_cluster(cluster3)
    assert_equal(len(clusters), 1)
    assert_array_equal(list(itertools.chain(*clusters)), list(cluster1))
    assert_array_equal(clusters.centroids, np.array([centroid1]))
    assert_equal(clusters[0], cluster1)

    clusters.remove_cluster(cluster1)
    assert_equal(len(clusters), 0)
    assert_array_equal(list(itertools.chain(*clusters)), [])
    assert_array_equal(clusters.centroids, [])


@set_random_number_generator(42)
def test_cluster_map_centroid_iter(rng):
    nb_clusters = 11

    cluster_map = ClusterMapCentroid()
    clusters = []
    for i in range(nb_clusters):
        new_centroid = np.zeros_like(features)
        new_cluster = ClusterCentroid(new_centroid,
                                      indices=rng.integers(0, len(data),
                                                           size=10))
        cluster_map.add_cluster(new_cluster)
        clusters.append(new_cluster)

    assert_true(all([c1 is c2 for c1, c2 in
                     zip(cluster_map.clusters, clusters)]))
    assert_array_equal(cluster_map, clusters)
    assert_array_equal(cluster_map.clusters, clusters)
    assert_array_equal(cluster_map, [cluster.indices for cluster in clusters])

    # Set refdata
    cluster_map.refdata = data
    for c1, c2 in zip(cluster_map, clusters):
        assert_arrays_equal(c1, [data[i] for i in c2.indices])


@set_random_number_generator()
def test_cluster_map_centroid_getitem(rng):
    nb_clusters = 11
    indices = list(range(len(data)))
    rng.shuffle(indices)  # None trivial ordering
    advanced_indices = indices + [0, 1, 2, -1, -2, -3]

    cluster_map = ClusterMapCentroid()
    clusters = []
    for i in range(nb_clusters):
        centroid = np.zeros_like(features)
        cluster = ClusterCentroid(centroid)
        cluster.id = cluster_map.add_cluster(cluster)
        clusters.append(cluster)

    # Test indexing
    for i in advanced_indices:
        assert_true(cluster_map[i] == clusters[i])

    # Test advanced indexing
    assert_arrays_equal(cluster_map[advanced_indices],
                        [clusters[i] for i in advanced_indices])

    # Test index out of bounds
    assert_raises(IndexError, cluster_map.__getitem__, len(clusters))
    assert_raises(IndexError, cluster_map.__getitem__, -len(clusters)-1)

    # Test slicing and negative indexing
    assert_true(cluster_map[-1] == clusters[-1])
    assert_array_equal(cluster_map[::2], clusters[::2])
    assert_arrays_equal(cluster_map[::-1], clusters[::-1])
    assert_arrays_equal(cluster_map[:-1], clusters[:-1])
    assert_arrays_equal(cluster_map[1:], clusters[1:])


def test_cluster_map_centroid_comparison_with_int():
    clusters1_indices = range(10)
    clusters2_indices = range(10, 15)
    clusters3_indices = [15]

    # Build a test ClusterMapCentroid
    centroid = np.zeros_like(features)
    cluster1 = ClusterCentroid(centroid.copy())
    for i in clusters1_indices:
        cluster1.assign(i, features)

    cluster2 = ClusterCentroid(centroid.copy())
    for i in clusters2_indices:
        cluster2.assign(i, features)

    cluster3 = ClusterCentroid(centroid.copy())
    for i in clusters3_indices:
        cluster3.assign(i, features)

    # Update centroids
    cluster1.update()
    cluster2.update()
    cluster3.update()

    clusters = ClusterMapCentroid()
    clusters.add_cluster(cluster1)
    clusters.add_cluster(cluster2)
    clusters.add_cluster(cluster3)

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


def test_subclassing_clustering():
    class SubClustering(Clustering):
        def cluster(self, data, ordering=None):
            pass

    clustering_algo = SubClustering()
    assert_raises(NotImplementedError,
                  super(SubClustering, clustering_algo).cluster,
                  None)
