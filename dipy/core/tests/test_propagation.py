import os
import numpy as np
import dipy as dp
from dipy.core.track_propagation import FACT_DeltaX
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
    #ten = dp.Tensor(data,bvals,gradients,thresh=50)
    
    t1=time()
    FD=dp.FACT_Delta(gqs.QA,gqs.IN,seeds_no=5)
    T1=FD.tracks
    t2=time()    
    print 'I', t2-t1, 'time.'
        
    T2=FACT_DeltaX(gqs.QA,gqs.IN,seed_list=FD.seed_list).tracks
    t3=time()    
    print 'X', t3-t2, 'time.'

    print (t3-t2)/(t2-t1), ' ratio '

    #T2=FACT_Delta2(ten.FA,ten.IN,seeds_no=1000,qa_thr=0.2).tracks
    
    print(len(T1))
    print(len(T2))

    print 'T1[0]'
    print T1[0]
    print 'T2[0]'
    print T2[0]

    print 'T1[1]'
    print T1[1]
    print 'T2[1]'
    print T2[1]  
