""" Benchmarks for vec / val summation routine

Run benchmarks with::

    import dipy.reconst as dire
    dire.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also
run the doctests, let's hope they pass.
"""
import numpy as np
from numpy.random import randn

from dipy.reconst.vec_val_sum import vec_val_vect

from dipy.utils.testing import measure
import pytest

try:
    np.einsum
    found = False
except AttributeError:
    found = True


@pytest.mark.skipif(found, reason="Need einsum for benchmark")
def bench_vec_val_vect():
    # nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench'
    repeat = 100
    etime = measure("np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)",
                    repeat)
    vtime = measure("vec_val_vect(evecs, evals)", repeat)
    print("einsum %4.2f; vec_val_vect %4.2f" % (etime, vtime))
