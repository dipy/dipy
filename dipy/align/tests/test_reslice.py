import nibabel as nib
import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_equal, assert_raises

from dipy.align.reslice import _lanczos_kernel, reslice
from dipy.data import get_fnames
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io.image import load_nifti


def test_resample():
    fimg, _, _ = get_fnames(name="small_25")
    data, affine, zooms = load_nifti(fimg, return_voxsize=True)

    # test that new zooms are correctly from the affine (check with 3D volume)
    new_zooms = (1, 1.2, 2.1)
    data2, affine2 = reslice(
        data[..., 0], affine, zooms, new_zooms, order=1, mode="constant"
    )
    img2 = nib.Nifti1Image(data2, affine2)
    new_zooms_confirmed = img2.header.get_zooms()[:3]
    assert_almost_equal(new_zooms, new_zooms_confirmed)

    # test that shape changes correctly for the first 3 dimensions (check 4D)
    new_zooms = (1, 1, 1.0)
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=0, mode="reflect")
    assert_equal(2 * np.array(data.shape[:3]), data2.shape[:3])
    assert_equal(data2.shape[-1], data.shape[-1])

    # same with different interpolation order
    new_zooms = (1, 1, 1.0)
    data3, affine2 = reslice(data, affine, zooms, new_zooms, order=5, mode="reflect")
    assert_equal(2 * np.array(data.shape[:3]), data3.shape[:3])
    assert_equal(data3.shape[-1], data.shape[-1])

    # test that the sigma will be reduced with interpolation
    sigmas = estimate_sigma(data)
    sigmas2 = estimate_sigma(data2)
    sigmas3 = estimate_sigma(data3)

    assert_(np.all(sigmas > sigmas2))
    assert_(np.all(sigmas2 > sigmas3))

    # check that 4D resampling matches 3D resampling
    data2, affine2 = reslice(data, affine, zooms, new_zooms)
    for i in range(data.shape[-1]):
        _data, _affine = reslice(data[..., i], affine, zooms, new_zooms)
        assert_almost_equal(data2[..., i], _data)
        assert_almost_equal(affine2, _affine)

    # check use of multiprocessing pool of specified size
    data3, affine3 = reslice(data, affine, zooms, new_zooms, num_processes=4)
    assert_almost_equal(data2, data3)
    assert_almost_equal(affine2, affine3)

    # check use of multiprocessing pool of autoconfigured size
    data3, affine3 = reslice(data, affine, zooms, new_zooms, num_processes=-1)
    assert_almost_equal(data2, data3)
    assert_almost_equal(affine2, affine3)

    # test invalid values of num_threads
    assert_raises(ValueError, reslice, data, affine, zooms, new_zooms, num_processes=0)

    # test invalid volume dimension
    assert_raises(
        ValueError, reslice, np.zeros((4, 4, 4, 4, 1)), affine, zooms, new_zooms
    )


def test_lanczos_kernel():
    assert_almost_equal(_lanczos_kernel(0.0, 2), 1.0)
    assert_equal(_lanczos_kernel(3.0, 2), 0.0)
    assert_equal(_lanczos_kernel(-3.0, 2), 0.0)
    vals = np.linspace(-1.9, 1.9, 20)
    assert_almost_equal(_lanczos_kernel(vals, 2), _lanczos_kernel(-vals, 2))


def test_reslice_lanczos():
    fimg, _, _ = get_fnames(name="small_25")
    data, affine, zooms = load_nifti(fimg, return_voxsize=True)

    new_zooms = (1.0, 1.0, 1.0)

    data_l2, affine_l2 = reslice(
        data[..., 0], affine, zooms, new_zooms, order="lanczos2"
    )
    data_l, affine_l = reslice(data[..., 0], affine, zooms, new_zooms, order="lanczos")
    assert_almost_equal(data_l2, data_l)
    assert_almost_equal(affine_l2, affine_l)

    data_lin, _ = reslice(data[..., 0], affine, zooms, new_zooms, order=1)
    assert_equal(data_l2.shape, data_lin.shape)

    data_l3, _ = reslice(data[..., 0], affine, zooms, new_zooms, order="lanczos3")
    assert_equal(data_l3.shape, data_lin.shape)

    data2_4d, _ = reslice(data, affine, zooms, new_zooms, order="lanczos2")
    assert_equal(data2_4d.shape[:3], data_l2.shape[:3])
    assert_equal(data2_4d.shape[-1], data.shape[-1])

    assert_almost_equal(affine_l2, affine_l)

    assert_raises(
        ValueError, reslice, data[..., 0], affine, zooms, new_zooms, order="bad"
    )
    assert_raises(ValueError, reslice, data[..., 0], affine, zooms, new_zooms, order=6)
    assert_raises(
        ValueError, reslice, data[..., 0], affine, zooms, new_zooms, mode="invalid"
    )
    assert_raises(
        ValueError,
        reslice,
        data[..., 0],
        affine,
        zooms,
        new_zooms,
        order="lanczos2",
        mode="mirror",
    )
