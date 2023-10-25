import numpy as np
import nibabel as nib
from numpy.testing import (assert_,
                           assert_equal,
                           assert_almost_equal,
                           assert_raises)
from dipy.io.image import load_nifti
from dipy.data import get_fnames
from dipy.align.reslice import reslice
from dipy.denoise.noise_estimate import estimate_sigma


def test_resample():
    fimg, _, _ = get_fnames("small_25")
    data, affine, zooms = load_nifti(fimg, return_voxsize=True)

    # test that new zooms are correctly from the affine (check with 3D volume)
    new_zooms = (1, 1.2, 2.1)
    data2, affine2 = reslice(data[..., 0], affine, zooms, new_zooms, order=1,
                             mode='constant')
    img2 = nib.Nifti1Image(data2, affine2)
    new_zooms_confirmed = img2.header.get_zooms()[:3]
    assert_almost_equal(new_zooms, new_zooms_confirmed)

    # test that shape changes correctly for the first 3 dimensions (check 4D)
    new_zooms = (1, 1, 1.)
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=0,
                             mode='reflect')
    assert_equal(2 * np.array(data.shape[:3]), data2.shape[:3])
    assert_equal(data2.shape[-1], data.shape[-1])

    # same with different interpolation order
    new_zooms = (1, 1, 1.)
    data3, affine2 = reslice(data, affine, zooms, new_zooms, order=5,
                             mode='reflect')
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
    assert_raises(ValueError, reslice, data, affine, zooms, new_zooms,
                  num_processes=0)

    # test invalid volume dimension
    assert_raises(ValueError, reslice, np.zeros((4, 4, 4, 4, 1)), affine,
                  zooms, new_zooms)
