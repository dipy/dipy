from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.testing as npt

from distutils.version import LooseVersion
from dipy.reconst.peaks import PeaksAndMetrics
from nibabel.tmpdirs import InTemporaryDirectory

# Conditional testing machinery for pytables
from dipy.testing import doctest_skip_parser

# Conditional import machinery for pytables
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests, if we don't have pytables
tables, have_tables, _ = optional_package('tables')

# Useful variable for backward compatibility.
if have_tables:
    TABLES_LESS_3_0 = LooseVersion(tables.__version__) < "3.0"

from dipy.data import get_sphere
from dipy.core.sphere import Sphere

from dipy.io.peaks import load_peaks, save_peaks


def test_io_peaks():

    with InTemporaryDirectory():

        fname = 'test.pam'

        sphere = get_sphere('repulsion724')

        pam = PeaksAndMetrics()
        pam.affine = np.eye(4)
        pam.peak_dirs = np.random.rand(10, 10, 10, 5, 3)
        pam.peak_values = np.zeros((10, 10, 10, 5))
        pam.peak_indices = np.zeros((10, 10, 10, 5))
        pam.shm_coeff = np.zeros((10, 10, 10, 45))
        pam.sphere = Sphere(xyz=sphere.vertices)
        pam.B = np.zeros((45, sphere.vertices.shape[0]))
        pam.total_weight = 0.5
        pam.ang_thr = 60
        pam.gfa = np.zeros((10, 10, 10))
        pam.qa = np.zeros((10, 10, 10, 5))
        pam.odf = np.zeros((10, 10, 10, sphere.vertices.shape[0]))

        save_peaks(fname, pam)
        pam2 = load_peaks(fname, verbose=True)
        npt.assert_array_equal(pam.peak_dirs, pam2.peak_dirs)

        pam2.affine = None

        fname2 = 'test2.pam'
        save_peaks(fname2, pam2)
        pam3 = load_peaks(fname2, verbose=True)






test_io_peaks()