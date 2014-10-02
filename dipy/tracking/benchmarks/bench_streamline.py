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

from dipy.tracking.streamline import set_number_of_points, length
from dipy.tracking.tests.test_streamline import set_number_of_points_python, length_python

from numpy.testing import measure


def bench_resample():
    repeat = 100
    nb_points_per_streamline = 100
    nb_points = 42
    nb_streamlines = int(1e4)
    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype("float32") for i in range(nb_streamlines)]

    print("Timing set_number_of_points() in Cython")
    cython_time = measure("set_number_of_points(streamlines, nb_points)", repeat)
    print("Cython time: {0:.3}sec".format(cython_time))

    # python_time = measure("[set_number_of_points_python(s, nb_points) for s in streamlines]", repeat)
    # print("Python time: {0:.2}sec".format(python_time))
    # print("Speed up of {0}x".format(python_time/cython_time))


def bench_length():
    repeat = 100
    nb_points_per_streamline = 100
    nb_streamlines = int(1e5)
    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype("float32") for i in range(nb_streamlines)]

    print("Timing length() in Cython")
    cython_time = measure("length(streamlines)", repeat)
    print("Cython time: {0:.3}sec".format(cython_time))

    # python_time = measure("[length_python(s) for s in streamlines]", repeat)
    # print("Python time: {0:.2}sec".format(python_time))
    # print("Speed up of {0}x".format(python_time/cython_time))


from dipy.tracking.streamlinespeed import set_number_of_points_old, length_old

def bench_length_and_resample_2():
    repeat = 1
    nb_points_per_streamline = 100
    nb_points = 42
    nb_streamlines = int(1e6)
    streamlines = [np.random.rand(nb_points_per_streamline, 3).astype("float32") for i in range(nb_streamlines)]

    print("Timing length() in Cython")
    new_time = measure("length(streamlines)", repeat)
    print("Cython time: {0:.3}sec".format(new_time))

    old_time = measure("length_old(streamlines)", repeat)
    print("Cython time: {0:.3}sec".format(old_time))
    print("Speed up of {0}x".format(old_time/new_time))

    print("")
    print("Timing set_number_of_points() in Cython")
    new_time = measure("set_number_of_points(streamlines, nb_points)", repeat)
    print("New version time: {0:.3}sec".format(new_time))

    old_time = measure("set_number_of_points_old(streamlines, nb_points)", repeat)
    print("Old version time: {0:.3}sec".format(old_time))
    print("Speed up of {0}x".format(old_time/new_time))
