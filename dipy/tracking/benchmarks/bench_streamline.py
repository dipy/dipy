""" Benchmarks for functions related to streamline

Run all benchmarks with::

    import dipy.tracking as dipytracking
    dipytracking.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also
run the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' bench_streamline.py
"""
import numpy as np
from numpy.testing import measure
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.data import get_data
from nibabel import trackvis as tv

from dipy.tracking.streamline import (set_number_of_points,
                                      length,
                                      compress_streamlines)
from dipy.tracking.tests.test_streamline import (set_number_of_points_python,
                                                 length_python,
                                                 compress_streamlines_python)

from dipy.tracking import Streamlines

DATA = {}


def setup():
    global DATA
    rng = np.random.RandomState(42)
    nb_streamlines = 20000
    min_nb_points = 2
    max_nb_points = 100

    DATA['rng'] = rng
    DATA['nb_streamlines'] = nb_streamlines
    DATA['streamlines'] = generate_streamlines(nb_streamlines,
                                               min_nb_points, max_nb_points,
                                               rng=rng)
    DATA['streamlines_arrseq'] = Streamlines(DATA['streamlines'])


def generate_streamlines(nb_streamlines, min_nb_points, max_nb_points, rng):
    streamlines = [rng.rand(*(rng.randint(min_nb_points, max_nb_points), 3))
                   for _ in range(nb_streamlines)]
    return streamlines


def bench_set_number_of_points():
    repeat = 5
    nb_points = 42
    nb_streamlines = DATA['nb_streamlines']
    streamlines = DATA["streamlines"]  # Streamlines as a list of ndarrays.

    msg = "Timing set_number_of_points() with {0:,} streamlines."
    print(msg.format(nb_streamlines * repeat))
    cython_time = measure("set_number_of_points(streamlines, nb_points)",
                          repeat)
    print("Cython time: {0:.3f} sec".format(cython_time))

    python_time = measure("[set_number_of_points_python(s, nb_points)"
                          " for s in streamlines]", repeat)
    print("Python time: {0:.2f} sec".format(python_time))
    print("Speed up of {0:.2f}x".format(python_time/cython_time))

    # Make sure it produces the same results.
    assert_array_almost_equal([set_number_of_points_python(s) for s in DATA["streamlines"]],
                              set_number_of_points(DATA["streamlines"]))

    streamlines = DATA['streamlines_arrseq']
    cython_time_arrseq = measure("set_number_of_points(streamlines, nb_points)", repeat)
    print("Cython time (ArrSeq): {0:.3f} sec".format(cython_time_arrseq))
    print("Speed up of {0:.2f}x".format(python_time/cython_time_arrseq))

    # Make sure it produces the same results.
    assert_array_equal(set_number_of_points(DATA["streamlines"]),
                       set_number_of_points(DATA["streamlines_arrseq"]))


def bench_length():
    repeat = 10
    nb_streamlines = DATA['nb_streamlines']
    streamlines = DATA["streamlines"]  # Streamlines as a list of ndarrays.

    msg = "Timing length() with {0:,} streamlines."
    print(msg.format(nb_streamlines * repeat))
    python_time = measure("[length_python(s) for s in streamlines]", repeat)
    print("Python time: {0:.2f} sec".format(python_time))

    cython_time = measure("length(streamlines)", repeat)
    print("Cython time: {0:.3f} sec".format(cython_time))
    print("Speed up of {0:.2f}x".format(python_time/cython_time))

    # Make sure it produces the same results.
    assert_array_almost_equal([length_python(s) for s in DATA["streamlines"]],
                              length(DATA["streamlines"]))

    streamlines = DATA['streamlines_arrseq']
    cython_time_arrseq = measure("length(streamlines)", repeat)
    print("Cython time (ArrSeq): {0:.3f} sec".format(cython_time_arrseq))
    print("Speed up of {0:.2f}x".format(python_time/cython_time_arrseq))

    # Make sure it produces the same results.
    assert_array_equal(length(DATA["streamlines"]),
                       length(DATA["streamlines_arrseq"]))


def bench_compress_streamlines():
    repeat = 10
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]

    print("Timing compress_streamlines() in Cython"
          " ({0} streamlines)".format(len(streamlines)))
    cython_time = measure("compress_streamlines(streamlines)", repeat)
    print("Cython time: {0:.3}sec".format(cython_time))
    del streamlines

    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    python_time = measure("map(compress_streamlines_python, streamlines)",
                          repeat)
    print("Python time: {0:.2}sec".format(python_time))
    print("Speed up of {0}x".format(python_time/cython_time))
    del streamlines
