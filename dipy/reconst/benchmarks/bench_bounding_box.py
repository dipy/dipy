from __future__ import division, print_function, absolute_import

""" Benchmarks for bounding_box

Run all benchmarks with::

    import dipy.reconst as dire
    dire.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also
run the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_bounding_box.py
"""
import numpy as np
from numpy.testing import measure
from dipy.segment.mask import bounding_box


def bench_bounding_box():
    vol = np.zeros((100, 100, 100))

    vol[0, 0, 0] = 1
    times = 100
    time = measure("bounding_box(vol)", times) / times
    print("Bounding_box on a sparse volume: {}".format(time))

    vol[:] = 10
    times = 1
    time = measure("bounding_box(vol)", times) / times
    print("Bounding_box on a dense volume: {}".format(time))

if __name__ == "__main__":
    bench_bounding_box()
