import numpy as np
import nose
import nibabel as nib
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data    
from dipy.reconst.dti import Tensor
from dipy.sims.phantom import orbital_phantom

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

    
    
if __name__ == "__main__":    
    test_phantom()

    
    
    
    
    
    
