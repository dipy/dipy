import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt

from dipy.direction.peaks import PeaksAndMetrics
from dipy.data import default_sphere
from dipy.io.peaks import load_peaks, save_peaks, peaks_to_niftis
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_io_peaks(rng):
    with TemporaryDirectory() as tmpdir:
        fname = 'test.pam5'

        pam = PeaksAndMetrics()
        pam.affine = np.eye(4)
        pam.peak_dirs = rng.random((10, 10, 10, 5, 3))
        pam.peak_values = np.zeros((10, 10, 10, 5))
        pam.peak_indices = np.zeros((10, 10, 10, 5))
        pam.shm_coeff = np.zeros((10, 10, 10, 45))
        pam.sphere = default_sphere
        pam.B = np.zeros((45, default_sphere.vertices.shape[0]))
        pam.total_weight = 0.5
        pam.ang_thr = 60
        pam.gfa = np.zeros((10, 10, 10))
        pam.qa = np.zeros((10, 10, 10, 5))
        pam.odf = np.zeros((10, 10, 10, default_sphere.vertices.shape[0]))

        save_peaks(pjoin(tmpdir, fname), pam)
        pam2 = load_peaks(pjoin(tmpdir, fname), verbose=True)
        npt.assert_array_equal(pam.peak_dirs, pam2.peak_dirs)

        pam2.affine = None

        fname2 = 'test2.pam5'
        save_peaks(pjoin(tmpdir, fname2), pam2, np.eye(4))
        pam2_res = load_peaks(pjoin(tmpdir, fname2), verbose=True)
        npt.assert_array_equal(pam.peak_dirs, pam2_res.peak_dirs)

        pam3 = load_peaks(pjoin(tmpdir, fname2), verbose=False)

        for attr in ['peak_dirs', 'peak_values', 'peak_indices',
                     'gfa', 'qa', 'shm_coeff', 'B', 'odf']:
            npt.assert_array_equal(getattr(pam3, attr),
                                   getattr(pam, attr))

        npt.assert_equal(pam3.total_weight, pam.total_weight)
        npt.assert_equal(pam3.ang_thr, pam.ang_thr)
        npt.assert_array_almost_equal(pam3.sphere.vertices,
                                      pam.sphere.vertices)

        fname3 = 'test3.pam5'
        pam4 = PeaksAndMetrics()
        npt.assert_raises((ValueError, AttributeError), save_peaks,
                          pjoin(tmpdir, fname3), pam4)

        fname4 = 'test4.pam5'
        del pam.affine
        save_peaks(pjoin(tmpdir, fname4), pam, affine=None)

        fname5 = 'test5.pkm'
        npt.assert_raises(OSError, save_peaks, pjoin(tmpdir, fname5), pam)

        pam.affine = np.eye(4)
        fname6 = 'test6.pam5'
        save_peaks(pjoin(tmpdir, fname6), pam, verbose=True)

        del pam.shm_coeff
        save_peaks(pjoin(tmpdir, fname6), pam, verbose=False)

        pam.shm_coeff = np.zeros((10, 10, 10, 45))
        del pam.odf
        save_peaks(pjoin(tmpdir, fname6), pam)
        pam_tmp = load_peaks(pjoin(tmpdir, fname6), True)
        npt.assert_equal(pam_tmp.odf, None)

        fname7 = 'test7.paw'
        npt.assert_raises(OSError, load_peaks, pjoin(tmpdir, fname7))

        del pam.shm_coeff
        save_peaks(pjoin(tmpdir, fname6), pam, verbose=True)

        fname_shm = 'shm.nii.gz'
        fname_dirs = 'dirs.nii.gz'
        fname_values = 'values.nii.gz'
        fname_indices = 'indices.nii.gz'
        fname_gfa = 'gfa.nii.gz'

        pam.shm_coeff = np.ones((10, 10, 10, 45))
        peaks_to_niftis(pam, fname_shm, fname_dirs, fname_values,
                        fname_indices, fname_gfa, reshape_dirs=False)

        os.path.isfile(fname_shm)
        os.path.isfile(fname_dirs)
        os.path.isfile(fname_values)
        os.path.isfile(fname_indices)
        os.path.isfile(fname_gfa)


@set_random_number_generator()
def test_io_save_peaks_error(rng):
    with TemporaryDirectory() as tmpdir:
        fname = 'test.pam5'

        pam = PeaksAndMetrics()

        npt.assert_raises(OSError, save_peaks, pjoin(tmpdir, 'test.pam'), pam)
        npt.assert_raises((ValueError, AttributeError), save_peaks,
                          pjoin(tmpdir, fname), pam)

        pam.affine = np.eye(4)
        pam.peak_dirs = rng.random((10, 10, 10, 5, 3))
        pam.peak_values = np.zeros((10, 10, 10, 5))
        pam.peak_indices = np.zeros((10, 10, 10, 5))
        pam.shm_coeff = np.zeros((10, 10, 10, 45))
        pam.sphere = default_sphere
        pam.B = np.zeros((45, default_sphere.vertices.shape[0]))
        pam.total_weight = 0.5
        pam.ang_thr = 60
        pam.gfa = np.zeros((10, 10, 10))
        pam.qa = np.zeros((10, 10, 10, 5))
        pam.odf = np.zeros((10, 10, 10, default_sphere.vertices.shape[0]))
