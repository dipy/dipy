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
    N=100 #timepoints

    # We make one with no noise, so that we can estimate SNR relative to it:
    vol = orbital_phantom(gtab,
                          func=f,
                          t=np.linspace(0,2*np.pi,N),
                          datashape=(10,10,10,len(bvals)),
                          origin=(5,5,5),
                          scale=(3,3,3),
                          angles=np.linspace(0,2*np.pi,16),
                          radii=np.linspace(0.2,2,6))

    for snr in [1, 10, 20]:
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

        s = np.mean(vol, -1)
        n = np.std(noise, -1)
        snr_est = s/n        
        assert_array_almost_equal(np.mean(snr_est[np.isfinite(snr_est)]),
                                  snr, decimal=0)


if __name__ == "__main__":
    #test_phantom()
    #test_noise()
    test_snr()


    
    
    
