import numpy as np
import nibabel as nib
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.data import get_data
from dipy.align.reslice import reslice


def test_resample():

    fimg, _, _ = get_data("small_25")

    img = nib.load(fimg)

    data = img.get_data()
    affine = img.get_affine()
    zooms = img.get_header().get_zooms[:3]

    print(affine)
    print(zooms)

    1/0
    new_zooms = ()

    data2, affine2 = reslice(data, affine, zooms, new_zooms, order=1,
                             mode='constant')





if __name__ == '__main__':

    run_module_suite()
