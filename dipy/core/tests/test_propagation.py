import os
import numpy as np
import dipy as dp
from dipy.core.track_propagation import FACT_DeltaX
from dipy.core.track_propagation_performance import ndarray_offset
import nibabel as ni
from os.path import join as opj
from time import time

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

def test_fact():
    #read bvals,gradients and data
    bvals=np.load(opj(os.path.dirname(__file__), \
                          'data','small_64D.bvals.npy'))
    gradients=np.load(opj(os.path.dirname(__file__), \
                              'data','small_64D.gradients.npy'))    
    img =ni.load(os.path.join(os.path.dirname(__file__),\
                                  'data','small_64D.nii'))
    data=img.get_data()    

    print(data.shape)    
    gqs = dp.GeneralizedQSampling(data,bvals,gradients)       
    ten = dp.Tensor(data,bvals,gradients,thresh=50)

    seed_list=np.dot(np.diag(np.arange(10)),np.ones((10,3)))
    
    T =FACT_DeltaX(gqs.QA,gqs.IN,seed_list=seed_list).tracks
    T2=FACT_DeltaX(ten.FA,ten.IN,seed_list=seed_list).tracks

    from dipy.core.track_metrics import length
    print('length T ',sum([length(t) for t in T]))  
    print('length T2',sum([length(t) for t in T2]))  

    print(gqs.QA[1,4,8,0])
    print(gqs.QA.ravel()[ndarray_offset(np.array([1,4,8,0]),np.array(gqs.QA.strides),4,8)])

    assert gqs.QA[1,4,8,0]==gqs.QA.ravel()[ndarray_offset(np.array([1,4,8,0]),np.array(gqs.QA.strides),4,8)]
    assert sum([length(t) for t in T])==78.
    assert sum([length(t) for t in T2])==63.50
