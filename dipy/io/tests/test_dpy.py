import os
import numpy as np

from nibabel.tmpdirs import InTemporaryDirectory

from dipy.io.dpy import Dpy

import numpy.testing as npt
from dipy.tracking.streamline import Streamlines


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
        dpw.write_tracks(Streamlines([C, B, A]))

        all_tracks = np.ascontiguousarray(np.vstack([A, B, C, C, B, A]))
        npt.assert_array_equal(all_tracks, dpw.tracks[:])
        dpw.close()

        dpr = Dpy(fname, 'r')
        npt.assert_equal(dpr.version() == u'0.0.1', True)
        T = dpr.read_tracksi([0, 1, 2, 0, 0, 2])
        T2 = dpr.read_tracks()
        npt.assert_equal(len(T2), 6)
        dpr.close()
        npt.assert_array_equal(A, T[0])
        npt.assert_array_equal(C, T[5])


if __name__ == '__main__':

    npt.run_module_suite()