import os
import numpy as np

from dipy.data import get_data
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dti import Tensor
from dipy.tracking.propagation import EuDX
from dipy.tracking.propspeed import ndarray_offset
from dipy.tracking.metrics import length

import nibabel as ni

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises, assert_almost_equal

from numpy.testing import assert_array_equal, assert_array_almost_equal

def test_eudx():
    
    #read bvals,gradients and data
    fimg,fbvals, fbvecs = get_data('small_64D')    
    bvals=np.load(fbvals)
    gradients=np.load(fbvecs)
    img =ni.load(fimg)    
    data=img.get_data()
    
    print(data.shape)    
    gqs = GeneralizedQSampling(data,bvals,gradients)       
    ten = Tensor(data,bvals,gradients,thresh=50)
    seed_list=np.dot(np.diag(np.arange(10)),np.ones((10,3)))    
    iT=iter(EuDX(gqs.qa(),gqs.ind(),seed_list=seed_list))
    T=[]
    for t in iT: 
        T.append(t)    
    iT2=iter(EuDX(ten.fa(),ten.ind(),seed_list=seed_list))
    T2=[]
    for t in iT2: 
        T2.append(t)
    
    print('length T ',sum([length(t) for t in T]))  
    print('length T2',sum([length(t) for t in T2]))  

    print(gqs.QA[1,4,8,0])
    print(gqs.QA.ravel()[ndarray_offset(np.array([1,4,8,0]),np.array(gqs.QA.strides),4,8)])

    assert_almost_equal(gqs.QA[1,4,8,0], gqs.QA.ravel()[ndarray_offset(np.array([1,4,8,0]),np.array(gqs.QA.strides),4,8)])

    #assert_equal, sum([length(t) for t in T ]) , 77.999996662139893
    #assert_equal, sum([length(t) for t in T2]) , 63.499998092651367
    assert_almost_equal(sum([length(t) for t in T ]) , 75.214988201856613)
    assert_almost_equal(sum([length(t) for t in T2]) , 60.202986091375351)


def uniform_seed_grid():

    #read bvals,gradients and data   
    fimg,fbvals, fbvecs = get_data('small_64D')    
    bvals=np.load(fbvals)
    gradients=np.load(fbvecs)
    img =ni.load(fimg)    
    data=img.get_data()
    
    x,y,z,g=data.shape   

    M=np.mgrid[.5:x-.5:np.complex(0,x),.5:y-.5:np.complex(0,y),.5:z-.5:np.complex(0,z)]
    M=M.reshape(3,x*y*z).T

    print(M.shape)
    print(M.dtype)

    for m in M: 
        print(m)
    gqs = GeneralizedQSampling(data,bvals,gradients)
    iT=iter(EuDX(gqs.QA,gqs.IN,seed_list=M))    
    T=[]
    for t in iT:
        T.append(i)
    
    print('lenT',len(T))
    assert_equal(len(T), 1221)



