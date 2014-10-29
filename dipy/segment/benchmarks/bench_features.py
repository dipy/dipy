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

from dipy.segment.metric import CenterOfMassFeature, IdentityFeature
from dipy.segment.featurespeed import extract

from numpy.testing import measure, assert_array_equal
from dipy.testing import assert_arrays_equal


def bench_extract():
    dtype = "float32"
    repeat = 10
    nb_points_per_streamline = 100
    nb_streamlines = int(1e4)

    feature = CenterOfMassFeature()

    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype(dtype) for i in range(nb_streamlines)]

    print("Timing extract() in Cython ({0} streamlines)".format(nb_streamlines))
    cython_time = measure("extract(feature, streamlines)", repeat)
    print("Purely Cython's time: {0:.3}sec".format(cython_time))
    del streamlines

    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype(dtype) for i in range(nb_streamlines)]
    cython_call_time = measure("[feature.extract(s) for s in streamlines]", repeat)
    print("Call to Cython's time: {0:.2}sec".format(cython_call_time))
    print("Speed up of {0}x".format(cython_call_time/cython_time))
    del streamlines

    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype(dtype) for i in range(nb_streamlines)]
    cython_result = extract(feature, streamlines)
    cython_calls_result = [feature.extract(s) for s in streamlines]
    assert_array_equal(cython_result, cython_calls_result)

    feature = IdentityFeature()

    streamlines = [np.random.rand(np.random.randint(1, nb_points_per_streamline), 3).astype(dtype) for i in range(nb_streamlines)]

    print("Timing extract() in Cython ({0} streamlines having different size)".format(nb_streamlines))
    cython_time = measure("extract(feature, streamlines)", repeat)
    print("Purely Cython's time: {0:.3}sec".format(cython_time))
    del streamlines

    streamlines = [np.random.rand(np.random.randint(1, nb_points_per_streamline), 3).astype(dtype) for i in range(nb_streamlines)]
    cython_call_time = measure("[feature.extract(s) for s in streamlines]", repeat)
    print("Call to Cython's time: {0:.2}sec".format(cython_call_time))
    print("Speed up of {0}x".format(cython_call_time/cython_time))
    del streamlines

    streamlines = [np.random.rand(np.random.randint(1, nb_points_per_streamline), 3).astype(dtype) for i in range(nb_streamlines)]
    cython_result = extract(feature, streamlines)
    cython_calls_result = [feature.extract(s) for s in streamlines]
    assert_arrays_equal(cython_result, cython_calls_result)
