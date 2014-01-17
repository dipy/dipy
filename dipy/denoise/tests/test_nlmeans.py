import numpy as np
from numpy.testing import (run_module_suite,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.denspeed import nlmeans_3d
from matplotlib.pyplot import *

def test_nlmeans():

    # A = 100 * np.zeros((50, 50, 50)) #+ 5 * np.random.rand(50, 50, 50)

    # assert_raises(ValueError, nlmeans_3d, A, sigma=5)

    # A = 100 + np.zeros((50, 50, 50)) #+ 5 * np.random.rand(50, 50, 50)

    # B = nlmeans_3d(A, sigma=1)#np.std(A))

    # assert_array_almost_equal(A, B, 2)

    # figure(1)

    # imshow(A[..., 25])

    # figure(2)

    # imshow(B[..., 25])

    import nibabel as nib
    vol = nib.load('t1.nii.gz')
    data = vol.get_data()[:, :, :, 0].astype('float64')
    aff = vol.get_affine()
    hdr = vol.get_header()

    print("vol size", data.shape)
    from time import time
    deb = time()
    den = nlmeans_3d(data, sigma=19.8849)
    print("total time", time()-deb)
    print("vol size", den.shape)
    nib.save(nib.Nifti1Image(den, aff, hdr), 't1_denoised.nii.gz')


test_nlmeans()
