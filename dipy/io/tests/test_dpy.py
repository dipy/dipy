import os
import numpy as np

from nibabel.tmpdirs import InTemporaryDirectory

from dipy.io.dpy import Dpy, have_tables


from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy.testing as npt

# Decorator to protect tests from being run without pytables present
iftables = npt.dec.skipif(not have_tables,
                          'Pytables does not appear to be installed')


@iftables
def test_dpy():
    fname = 'test.bin'
    with InTemporaryDirectory():
        dpw = Dpy(fname, 'w')
        A = np.ones((5, 3))
        B = 2 * A.copy()
        C = 3 * A.copy()
        dpw.write_track(A)
        dpw.write_track(B)
        dpw.write_track(C)
        dpw.write_tracks([C, B, A])
        dpw.close()
        dpr = Dpy(fname, 'r')
        assert_equal(dpr.version() == '0.0.1', True)
        T = dpr.read_tracksi([0, 1, 2, 0, 0, 2])
        T2 = dpr.read_tracks()
        assert_equal(len(T2), 6)
        dpr.close()
        assert_array_equal(A, T[0])
        assert_array_equal(C, T[5])
