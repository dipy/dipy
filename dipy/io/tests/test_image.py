from __future__ import division, print_function, absolute_import

import numpy.testing as npt
import nibabel as nib
from dipy.data import get_data
from dipy.io.image import load_nifti

def test_load():
    data_path, _, _ = get_data('small_25')
    nib_img = nib.load(data_path)
    nib_dat = nib_img.get_data()
    nib_vox_size = nib_img.get_header().get_zooms()[:3]
    nib_affine = nib_img.get_affine()

    print('No vox_size, no img')
    data, affine = load_nifti(data_path)

    npt.assert_array_equal(data, nib_dat)
    npt.assert_array_equal(affine, nib_affine)

    print('Vox_size, no img')
    data, affine, vox_size = load_nifti(data_path, return_voxsize=True)
    npt.assert_array_equal(data, nib_dat)
    npt.assert_array_equal(affine, nib_affine)
    npt.assert_array_equal(vox_size, nib_vox_size)

    print('Vox_size, Img')
    data, affine, vox_size, img = load_nifti(data_path, return_voxsize=True,
                                             return_img=True)
    npt.assert_array_equal(data, nib_dat)
    npt.assert_array_equal(affine, nib_affine)
    npt.assert_array_equal(vox_size, nib_vox_size)

    npt.assert_array_equal(nib_img.get_header().values, img.get_header().values)
    npt.assert_array_equal(nib_img.get_header().keys, img.get_header().keys)

if __name__ == '__main__':
    test_load()
