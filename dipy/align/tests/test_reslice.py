import numpy as np
import nibabel as nib
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_almost_equal)
from dipy.data import get_data
from dipy.align.reslice import reslice
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.align.aniso2iso import resample


def test_resample():

    fimg, _, _ = get_data("small_25")

    img = nib.load(fimg)

    data = img.get_data()
    affine = img.get_affine()
    zooms = img.get_header().get_zooms()[:3]

    # test that new zooms are correctly from the affine (check with 3D volume)
    new_zooms = (1, 1.2, 2.1)

    data2, affine2 = reslice(data[..., 0], affine, zooms, new_zooms, order=1,
                             mode='constant')

    img2 = nib.Nifti1Image(data2, affine2)
    new_zooms_confirmed = img2.get_header().get_zooms()[:3]

    assert_almost_equal(new_zooms, new_zooms_confirmed)

    # same with resample
    new_zooms = (1, 1.2, 2.1)

    data2, affine2 = resample(data[..., 0], affine, zooms, new_zooms, order=1,
                              mode='constant')

    img2 = nib.Nifti1Image(data2, affine2)
    new_zooms_confirmed = img2.get_header().get_zooms()[:3]

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


if __name__ == '__main__':

    run_module_suite()