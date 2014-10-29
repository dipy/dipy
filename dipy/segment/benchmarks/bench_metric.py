""" Benchmarks for QuickBundles

Run all benchmarks with::

    import dipy.segment as dipysegment
    dipysegment.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_quickbundles.py
"""
import numpy as np

from dipy.segment.metric import MinimumAverageDirectFlipMetric as MDF
from dipy.segment.metricspeed import distance_matrix
from dipy.segment.metricspeed import dist

from numpy.testing import measure, assert_array_equal
from dipy.testing import assert_arrays_equal


def distance_matrix_python(metric, streamlines1, streamlines2):
    distances = np.zeros((len(streamlines1), len(streamlines2)))
    for i, s1 in enumerate(streamlines1):
        for j, s2 in enumerate(streamlines2):
            distances[i, j] = dist(metric, s1, s2)

    return distances


def bench_distance_metric():
    dtype = "float32"
    repeat = 1
    nb_points_per_streamline = 100
    nb_streamlines1 = 1130
    nb_streamlines2 = 770

    metric = MDF()

    streamlines1 = [np.random.rand(nb_points_per_streamline, 3).astype(dtype) for i in range(nb_streamlines1)]
    streamlines2 = [np.random.rand(nb_points_per_streamline, 3).astype(dtype) for i in range(nb_streamlines2)]

    print("Timing distance_matrix() in Cython ({0}x{1} streamlines)".format(nb_streamlines1, nb_streamlines2))
    cython_time = measure("distance_matrix(metric, streamlines1, streamlines2)", repeat)
    print("Purely Cython's time: {0:.3f}sec".format(cython_time))
    del streamlines1
    del streamlines2

    streamlines1 = [np.random.rand(nb_points_per_streamline, 3).astype(dtype) for i in range(nb_streamlines1)]
    streamlines2 = [np.random.rand(nb_points_per_streamline, 3).astype(dtype) for i in range(nb_streamlines2)]
    cython_call_time = measure("distance_matrix_python(metric, streamlines1, streamlines2)", repeat)
    print("Call to Cython's time: {0:.3f}sec".format(cython_call_time))
    print("Speed up of {0}x".format(cython_call_time/cython_time))
    #del streamlines1
    #del streamlines2

    cython_result = distance_matrix(metric, streamlines1, streamlines2)
    cython_calls_result = distance_matrix_python(metric, streamlines1, streamlines2)
    assert_array_equal(cython_result, cython_calls_result)
