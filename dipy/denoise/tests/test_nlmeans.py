import numpy as np
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.denspeed import nlmeans_3d
from matplotlib.pyplot import *


def test_nlmeans_random_noise():
    S0 = 100 + 5 * np.random.standard_normal((50, 50, 50))

    S0 = S0.astype('f8')

    from time import time

    t1 = time()
    S0n = nlmeans_3d(S0, sigma = 5, rician=False)
    t2 = time()
    print('Time was', t2 - t1)

    print(S0.mean(), S0.min(), S0.max())
    print(S0n.mean(), S0n.min(), S0n.max())

    assert_(S0n.min() > S0.min())

def test_nlmeans():

    import nibabel as nib
    vol = nib.load('/home/eleftherios/Desktop/t1.nii.gz')
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


#test_nlmeans_random_noise()
test_nlmeans()
