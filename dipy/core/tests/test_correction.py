import numpy as np
import dipy as dp
import nibabel as ni

import dipy.core.correction as corr

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

#from dipy.testing import parametric

#@parametric

def test_default_motion_correction():

    #create synthetic data
    S0=255*np.ones((50,50,50))#.astype('uint16')
    print S0.shape    
    S0=corr.add_padding(S0,5,125)
    print S0.shape  
    S0=corr.add_padding(S0,5)
    print S0.shape

    print np.sum(S0==255)
    print np.sum(S0==125)
    print np.sum(S0==0)
    
    A=np.array([0,0,0,.11,.12,.13,0,0,0,0,0,0])
    A=dp._affine(A)

    S0img=ni.Nifti1Image(S0,np.eye(4))
    S1img,A2=dp.volume_transform(S0img, A, reference=S0img,interp_order=0), A
    S1=S1img.get_data()

    S=np.zeros(S0.shape+(2,))
    S[:,:,:,0]=S0
    S[:,:,:,1]=S1
   
    
    #correct them
    S_corr,mats=corr.motion_correction(S,np.eye(4),ref=0,similarity='mi',subsampling=[1,1,1],interp='tri',order=0)

    print S.shape
    print S_corr.shape
    
    Simg=ni.Nifti1Image(S,np.eye(4))
    #ni.save(Simg,'/tmp/Simg.nii.gz')   
    Simg2=ni.Nifti1Image(S_corr,np.eye(4))
    #ni.save(Simg2,'/tmp/Simg2.nii.gz')

    #yield assert_array_almost_equal(xyz, pt)
    #return S,S_corr,mats


