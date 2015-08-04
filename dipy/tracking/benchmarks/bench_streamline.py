""" Benchmarks for functions related to streamline

Run all benchmarks with::

    import dipy.tracking as dipytracking
    dipytracking.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_streamline.py
"""
import numpy as np
from numpy.testing import measure
from dipy.data import get_data
from nibabel import trackvis as tv

from dipy.tracking.streamline import (set_number_of_points,
                                      length,
                                      compress_streamlines)
from dipy.tracking.tests.test_streamline import (set_number_of_points_python,
                                                 length_python,
                                                 compress_streamlines_python)


def bench_set_number_of_points():
    repeat = 1
    nb_points_per_streamline = 100
    nb_points = 42
    nb_streamlines = int(1e4)
    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype("float32") for i in range(nb_streamlines)]

    print("Timing set_number_of_points() in Cython ({0} streamlines)".format(nb_streamlines))
    cython_time = measure("set_number_of_points(streamlines, nb_points)", repeat)
    print("Cython time: {0:.3}sec".format(cython_time))
    del streamlines

    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype("float32") for i in range(nb_streamlines)]
    python_time = measure("[set_number_of_points_python(s, nb_points) for s in streamlines]", repeat)
    print("Python time: {0:.2}sec".format(python_time))
    print("Speed up of {0}x".format(python_time/cython_time))
    del streamlines


def bench_length():
    repeat = 1
    nb_points_per_streamline = 100
    nb_streamlines = int(1e5)
    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype("float32") for i in range(nb_streamlines)]

    print("Timing length() in Cython ({0} streamlines)".format(nb_streamlines))
    cython_time = measure("length(streamlines)", repeat)
    print("Cython time: {0:.3}sec".format(cython_time))
    del streamlines

    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype("float32") for i in range(nb_streamlines)]
    python_time = measure("[length_python(s) for s in streamlines]", repeat)
    print("Python time: {0:.2}sec".format(python_time))
    print("Speed up of {0}x".format(python_time/cython_time))
    del streamlines


def bench_compress_streamlines():
    repeat = 10
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]

    print("Timing compress_streamlines() in Cython ({0} streamlines)".format(len(streamlines)))
    cython_time = measure("compress_streamlines(streamlines)", repeat)
    print("Cython time: {0:.3}sec".format(cython_time))
    del streamlines

    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    python_time = measure("map(compress_streamlines_python, streamlines)", repeat)
    print("Python time: {0:.2}sec".format(python_time))
    print("Speed up of {0}x".format(python_time/cython_time))
    del streamlines
