import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel as nib
from dipy.data import get_data, get_sphere
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrumImaging

def test_dandelion():
    
    fimg,fbvals,fbvecs=get_data('small_101D')    
    bvals=np.loadtxt(fbvals)
    gradients=np.loadtxt(fbvecs).T
    data=nib.load(fimg).get_data()    
    
    """
    print(bvals.shape, gradients.shape, data.shape)    
    sd=SphericalDandelion(data,bvals,gradients)    
    
    sdf=sd.spherical_diffusivity(data[5,5,5])    
    
    XA=sd.xa()
    np.set_printoptions(2)
    print XA.min(),XA.max(),XA.mean()
    print sdf*10**4
    """
    
    










