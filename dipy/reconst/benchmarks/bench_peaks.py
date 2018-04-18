""" Benchmarks for peak finding

Run all benchmarks with::

    import dipy.reconst as dire
    dire.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also
run the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_peaks.py
"""
import numpy as np

from dipy.reconst.recspeed import local_maxima
from dipy.data import get_sphere
from dipy.core.sphere import unique_edges

from numpy.testing import measure


def bench_local_maxima():
    repeat = 10000
    sphere = get_sphere('symmetric724')
    vertices, faces = sphere.vertices, sphere.faces
    print('Timing peak finding')
    timed0 = measure("local_maxima(odf, edges)", repeat)
    print('Actual sphere: %0.2f' % timed0)
    # Create an artificial odf with a few peaks
    odf = np.zeros(len(vertices))
    odf[1] = 1.
    odf[143] = 143.
    odf[505] = 505.
    timed1 = measure("local_maxima(odf, edges)", repeat)
    print('Few-peak sphere: %0.2f' % timed1)
