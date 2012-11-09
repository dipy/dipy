import numpy as np
import nose
import nibabel as nib
import numpy.testing.decorators as dec

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_almost_equal)

from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data    
from dipy.reconst.dti import Tensor
from dipy.sims.phantom import orbital_phantom, add_noise
from dipy.core.gradients import gradient_table


def f(t):
    """
    Helper function used to define a mapping time => xyz
    """
    x=np.sin(t)
    y=np.cos(t)
    z=np.linspace(-1,1,len(x))
    return x,y,z

def test_phantom():
    

    fimg,fbvals,fbvecs=get_data('small_64D')    
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    bvecs[np.isnan(bvecs)]=0

    gtab = gradient_table(bvals, bvecs)
    
    N=50 #timepoints
    
    vol=orbital_phantom(gtab,
                        func=f,
                        t=np.linspace(0,2*np.pi,N),
                        datashape=(10,10,10,len(bvals)),
                        origin=(5,5,5),
                        scale=(3,3,3),
                        angles=np.linspace(0,2*np.pi,16),
                        radii=np.linspace(0.2,2,6))
    
    ten=Tensor(vol,bvals,bvecs)
    FA=ten.fa()
    FA[np.isnan(FA)]=0
    
    assert_equal(np.round(FA.max()*1000),707)

def test_noise():
    """
    Test that the noise added to the volume has the right variance
    """
    # The test should pass at the limit of (sufficiently) high signal, where
    # sigma is the std of the Rician:
    a = 100
    b = 100
    # Make a uniformly-distributed signal in a 4D volume. For this to
    # work, we probably need the last dimension to be rather large.
    vol = np.random.rand(10,10,10,100) * a + b
    for sigma in [0.1, 1, 10]:
        for noise_type in ['gaussian', 'rician']:
            vol_w_noise = add_noise(vol,
                                    sigma,
                                    noise_type=noise_type)
            noise = vol_w_noise - vol
            assert_array_almost_equal(np.std(noise), sigma, decimal=1)

@dec.slow
def test_snr():
    """
    Test the addition of noise to a phantom.

    """

    fimg,fbvals,fbvecs=get_data('small_64D')
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    bvecs[np.isnan(bvecs)]=0
    gtab = gradient_table(bvals, bvecs)
    N=50 #timepoints

    # We make one with no noise, so that we can estimate SNR relative to it:
    vol = orbital_phantom(gtab,
                          func=f,
                          t=np.linspace(0,2*np.pi,N),
                          datashape=(10,10,10,len(bvals)),
                          origin=(5,5,5),
                          scale=(3,3,3),
                          angles=np.linspace(0,2*np.pi,16),
                          radii=np.linspace(0.2,2,6))

    mean_sig = np.mean(vol)

    for snr in [20, 200]:
        vol_w_noise=orbital_phantom(gtab,
                                    func=f,
                                    t=np.linspace(0,2*np.pi,N),
                                    datashape=(10,10,10,len(bvals)),
                                    origin=(5,5,5),
                                    scale=(3,3,3),
                                    angles=np.linspace(0,2*np.pi,16),
                                    radii=np.linspace(0.2,2,6),
                                    snr=snr)

        noise = vol - vol_w_noise
        rms_noise = np.sqrt(np.mean(noise**2))
        snr_est = mean_sig/rms_noise
        assert_array_almost_equal(snr_est, snr, decimal=3)



if __name__ == "__main__":
    #test_phantom()
    #test_noise()
    test_snr()


    
    
    
