import os
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.gradients import gradient_table
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.data import default_sphere, get_fnames
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.image import load_nifti
from dipy.io.peaks import (
    load_pam,
    load_peaks,
    niftis_to_pam,
    pam_to_niftis,
    save_pam,
    save_peaks,
    tensor_to_pam,
)
import dipy.reconst.dti as dti
from dipy.testing.decorators import set_random_number_generator


def generate_default_pam(rng):
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
    return pam


@set_random_number_generator()
def test_io_peaks(rng):
    with TemporaryDirectory() as tmpdir:
        fname = pjoin(tmpdir, "test.pam5")

        pam = generate_default_pam(rng)
        save_pam(fname, pam)
        pam2 = load_pam(fname, verbose=False)
        npt.assert_array_equal(pam.peak_dirs, pam2.peak_dirs)

        pam2.affine = None

        fname2 = pjoin(tmpdir, "test2.pam5")
        save_pam(fname2, pam2, affine=np.eye(4))
        pam2_res = load_pam(fname2, verbose=True)
        npt.assert_array_equal(pam.peak_dirs, pam2_res.peak_dirs)

        pam3 = load_pam(fname2, verbose=False)

        for attr in [
            "peak_dirs",
            "peak_values",
            "peak_indices",
            "gfa",
            "qa",
            "shm_coeff",
            "B",
            "odf",
        ]:
            npt.assert_array_equal(getattr(pam3, attr), getattr(pam, attr))

        npt.assert_equal(pam3.total_weight, pam.total_weight)
        npt.assert_equal(pam3.ang_thr, pam.ang_thr)
        npt.assert_array_almost_equal(pam3.sphere.vertices, pam.sphere.vertices)

        fname3 = pjoin(tmpdir, "test3.pam5")
        pam4 = PeaksAndMetrics()
        npt.assert_raises((ValueError, AttributeError), save_pam, fname3, pam4)

        fname4 = pjoin(tmpdir, "test4.pam5")
        del pam.affine
        save_pam(fname4, pam, affine=None)

        fname5 = pjoin(tmpdir, "test5.pkm")
        npt.assert_raises(IOError, save_pam, fname5, pam)

        pam.affine = np.eye(4)
        fname6 = pjoin(tmpdir, "test6.pam5")
        save_pam(fname6, pam, verbose=True)

        del pam.shm_coeff
        save_pam(pjoin(tmpdir, fname6), pam, verbose=False)

        pam.shm_coeff = np.zeros((10, 10, 10, 45))
        del pam.odf
        save_pam(fname6, pam)
        pam_tmp = load_pam(fname6, verbose=True)
        npt.assert_equal(pam_tmp.odf, None)

        fname7 = pjoin(tmpdir, "test7.paw")
        npt.assert_raises(OSError, load_pam, fname7)

        del pam.shm_coeff
        save_pam(fname6, pam, verbose=True)

        fname_shm = pjoin(tmpdir, "shm.nii.gz")
        fname_dirs = pjoin(tmpdir, "peaks_dirs.nii.gz")
        fname_values = pjoin(tmpdir, "peaks_values.nii.gz")
        fname_indices = pjoin(tmpdir, "peaks_indices.nii.gz")
        fname_gfa = pjoin(tmpdir, "gfa.nii.gz")
        fname_sphere = pjoin(tmpdir, "sphere.txt")
        fname_b = pjoin(tmpdir, "B.nii.gz")
        fname_qa = pjoin(tmpdir, "qa.nii.gz")

        pam.shm_coeff = np.ones((10, 10, 10, 45))
        pam_to_niftis(
            pam,
            fname_shm=fname_shm,
            fname_peaks_dir=fname_dirs,
            fname_peaks_values=fname_values,
            fname_peaks_indices=fname_indices,
            fname_gfa=fname_gfa,
            fname_sphere=fname_sphere,
            fname_b=fname_b,
            fname_qa=fname_qa,
            reshape_dirs=False,
        )

        for name in [
            "shm.nii.gz",
            "peaks_dirs.nii.gz",
            "peaks_values.nii.gz",
            "gfa.nii.gz",
            "peaks_indices.nii.gz",
            "shm.nii.gz",
        ]:
            npt.assert_(
                os.path.isfile(pjoin(tmpdir, name)),
                "{} file does not exist".format(pjoin(tmpdir, name)),
            )


@set_random_number_generator()
def test_io_save_pam_error(rng):
    with TemporaryDirectory() as tmpdir:
        fname = "test.pam5"

        pam = PeaksAndMetrics()

        npt.assert_raises(IOError, save_pam, pjoin(tmpdir, "test.pam"), pam)
        npt.assert_raises(
            (ValueError, AttributeError), save_pam, pjoin(tmpdir, fname), pam
        )

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


def test_io_niftis_to_pam():
    with TemporaryDirectory() as tmpdir:
        pam = niftis_to_pam(
            affine=np.eye(4),
            peak_dirs=np.random.rand(10, 10, 10, 5, 3),
            peak_values=np.zeros((10, 10, 10, 5)),
            peak_indices=np.zeros((10, 10, 10, 5)),
            shm_coeff=np.zeros((10, 10, 10, 45)),
            sphere=default_sphere,
            gfa=np.zeros((10, 10, 10)),
            B=np.zeros((45, default_sphere.vertices.shape[0])),
            qa=np.zeros((10, 10, 10, 5)),
            odf=np.zeros((10, 10, 10, default_sphere.vertices.shape[0])),
            total_weight=0.5,
            ang_thr=60,
            pam_file=pjoin(tmpdir, "test15.pam5"),
        )

        npt.assert_equal(pam.peak_dirs.shape, (10, 10, 10, 5, 3))
        npt.assert_(os.path.isfile(pjoin(tmpdir, "test15.pam5")))


def test_tensor_to_pam():
    fdata, fbval, fbvec = get_fnames(name="small_25")
    gtab = gradient_table(fbval, bvecs=fbvec)
    data, affine = load_nifti(fdata)
    dm = dti.TensorModel(gtab)
    df = dm.fit(data)
    df.evals[0, 0, 0] = np.array([0, 0, 0])
    sphere = create_unit_sphere(recursion_level=4)
    odf = df.odf(sphere)

    with TemporaryDirectory() as tmpdir:
        fname = "test_tt.pam5"
        pam = tensor_to_pam(
            evals=df.evals,
            evecs=df.evecs,
            affine=affine,
            sphere=sphere,
            odf=odf,
            pam_file=pjoin(tmpdir, fname),
        )
        npt.assert_(os.path.isfile(pjoin(tmpdir, fname)))
        save_pam(pjoin(tmpdir, "test_tt_2.pam5"), pam)
        pam2 = load_pam(pjoin(tmpdir, "test_tt_2.pam5"))

        npt.assert_array_equal(pam.peak_values, pam2.peak_values)
        npt.assert_array_equal(pam.peak_dirs, pam2.peak_dirs)
        npt.assert_array_almost_equal(pam.peak_indices, pam2.peak_indices)
        del pam


@set_random_number_generator()
def test_io_peaks_deprecated(rng):
    with TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings(record=True) as cw:
            warnings.simplefilter("always", DeprecationWarning)
            fname = pjoin(tmpdir, "test_tt.pam5")
            pam = generate_default_pam(rng)
            save_peaks(fname, pam)
            pam2 = load_peaks(fname, verbose=True)
            npt.assert_array_equal(pam.peak_dirs, pam2.peak_dirs)
            npt.assert_equal(len(cw), 2)
            npt.assert_(issubclass(cw[0].category, DeprecationWarning))
            npt.assert_(issubclass(cw[1].category, DeprecationWarning))
