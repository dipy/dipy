import os
import numpy as np
import dipy as dp
from dipy.core.track_propagation import FACT_Delta2
import nibabel as ni
from os.path import join as opj

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
        
    T1=FACT_Delta2(gqs.QA,gqs.IN,seeds_no=1000).tracks
    T2=FACT_Delta2(ten.FA,ten.IN,seeds_no=1000,qa_thr=0.2).tracks
    
    print(len(T1))
    print(len(T2))
