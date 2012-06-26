import numpy as np
import nose
import nibabel as nib

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_almost_equal)

from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data    
from dipy.reconst.dti import Tensor
from dipy.sims.phantom import orbital_phantom, add_noise


def test_phantom():
    
    def f(t):
        x=np.sin(t)
        y=np.cos(t)    
        z=np.linspace(-1,1,len(x))
        return x,y,z

    fimg,fbvals,fbvecs=get_data('small_64D')    
    bvals=np.load(fbvals)
    bvecs=np.load(fbvecs)
    bvecs[np.isnan(bvecs)]=0
    
    N=50 #timepoints
    
    vol=orbital_phantom(bvals=bvals,
                         bvecs=bvecs,
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
    Test that the noise added to the volume has the right SNR
    """

    # np.random.seed(1977)

    # Make a uniformly-distributed signal in a 4D volume. For this to work, the
    # last dimension of the volume needs to be rather long:
    vol = np.random.rand(4,4,4,10e4) * 100 + 100
    sig_power = np.var(vol,axis=-1)


    for SNR in [0.1, 1, 10, 100]:
        for noise_type in ['gaussian', 'rician']:
            print noise_type
            vol_w_noise = add_noise(vol, SNR, noise_type=noise_type)
            noise = vol_w_noise - vol
            est_SNR = np.mean(vol)/np.std(noise)
            # And tolerance needs to be pretty lax...
            assert_array_almost_equal(est_SNR, SNR, decimal=2)


if __name__ == "__main__":
    test_phantom()
    test_noise()



    
    
    
