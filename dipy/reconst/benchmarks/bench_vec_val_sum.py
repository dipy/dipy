""" Benchmarks for vec / val summation routine

Run benchmarks with::

    import dipy.reconst as dire
    dire.bench()

With Pytest, Run this benchmark with:

    pytest -svv -c bench.ini /path/to/bench_vec_val_sum.py
"""
import numpy as np
from numpy.random import randn

from dipy.reconst.vec_val_sum import vec_val_vect

from numpy.testing import measure, dec

try:
    np.einsum
except AttributeError:
    with_einsum = dec.skipif(True, "Need einsum for benchmark")
else:
    def with_einsum(f): return f


@with_einsum
def bench_vec_val_vect():
    repeat = 100
    etime = measure("np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)",
                    repeat)
    vtime = measure("vec_val_vect(evecs, evals)", repeat)
    print("einsum %4.2f; vec_val_vect %4.2f" % (etime, vtime))
