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
    iT=iter(EuDX(gqs.qa(),gqs.ind(),seeds=seed_list))
    T=[]
    for t in iT: 
        T.append(t)    
    iT2=iter(EuDX(ten.fa(),ten.ind(),seeds=seed_list))
    T2=[]
    for t in iT2: 
        T2.append(t)
    
    print('length T ',sum([length(t) for t in T]))  
    print('length T2',sum([length(t) for t in T2]))  

    print(gqs.QA[1,4,8,0])
    print(gqs.QA.ravel()[ndarray_offset(np.array([1,4,8,0]),np.array(gqs.QA.strides),4,8)])

    assert_almost_equal(gqs.QA[1,4,8,0], gqs.QA.ravel()[ndarray_offset(np.array([1,4,8,0]),np.array(gqs.QA.strides),4,8)])

    assert_almost_equal(sum([length(t) for t in T ]) , 70.999996185302734,places=3)
    assert_almost_equal(sum([length(t) for t in T2]) , 56.999997615814209,places=3)


def test_eudx_further():
    """ Cause we love testin.. ;-)
    """

    fimg,fbvals,fbvecs=get_data('small_101D')
    img=ni.load(fimg)
    affine=img.get_affine()
    bvals=np.loadtxt(fbvals)
    gradients=np.loadtxt(fbvecs).T    
    data=img.get_data()
    ten=Tensor(data,bvals,gradients,thresh=50)
    x,y,z=data.shape[:3]
    seeds=np.zeros((10**4,3))
    for i in range(10**4):
        rx=(x-1)*np.random.rand()
        ry=(y-1)*np.random.rand()
        rz=(z-1)*np.random.rand()            
        seeds[i]=np.ascontiguousarray(np.array([rx,ry,rz]),dtype=np.float64)
    
    #print seeds
    #"""    
    eu=EuDX(a=ten.fa(),ind=ten.ind(),seeds=seeds,a_low=.2)
    T=[e for e in eu]
    
    #check that there are no negative elements
    for t in T:
        assert_equal(np.sum(t.ravel()<0),0)
    
    
    """
    
    for (i,t) in enumerate(T):
        for row in t:
            if row[0]<0 or row[1]<0 or row[2]<0:
                print 'l======'
                print i,row
                print t[0]
                print t[-1]
                
            if row[0]>=data.shape[0] or row[1]>=data.shape[1] or row[2]>=data.shape[2]:
                print 'h======'
                print i,row
                print t[0]
                print t[-1]
            
    
    from dipy.viz import fvtk
    
    r=fvtk.ren()
    fvtk.add(r,fvtk.line(T,fvtk.red))
    fvtk.add(r,fvtk.point(seeds,fvtk.green))
    fvtk.show(r)
    """

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
    iT=iter(EuDX(gqs.QA,gqs.IN,seeds=M))    
    T=[]
    for t in iT:
        T.append(i)
    
    print('lenT',len(T))
    assert_equal(len(T), 1221)



