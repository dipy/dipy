import numpy as np
import dipy as dp
import nibabel as ni
from scipy.ndimage import rotate
from nipy.neurospin.registration import register, transform


import dipy.core.correction as corr

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_default_motion_correction():

    #create synthetic data
    S0=200*np.ones((50,50,50))#.astype('uint16')
    S0=corr.add_padding(S0,10,100)
    S0=corr.add_padding(S0,40)

    #rotate second volume
    #A=np.array([0,0,0,.4,.4,.4,0,0,0,0,0,0])   
    #A=dp._affine(A)
    
    S0=rotate(S0,5,reshape=False)
    
    S1=rotate(S0,30,reshape=False)
    S=np.zeros(S0.shape+(2,))
 
    S0img=ni.Nifti1Image(S0,np.eye(4))
    ni.save(S0img,'/tmp/S0img.nii.gz')

    S1img=ni.Nifti1Image(S1,np.eye(4))
    ni.save(S1img,'/tmp/S1img.nii.gz')

    T=register(S1img,S0img,interp='pv')
    NS1img=transform(S1img, T)
    ni.save(NS1img,'/tmp/NS1img.nii.gz')
    

    '''
    Simg=ni.Nifti1Image(S,np.eye(4))
    ni.save(Simg,'/tmp/Simg.nii.gz')  

    S_corr,mats=corr.motion_correction(S,np.eye(4),ref=0,similarity='cr',subsampling=[1,1,1],interp='tri',order=3)

    print S.shape
    print S_corr.shape

    Simg2=ni.Nifti1Image(S_corr,np.eye(4))
    ni.save(Simg2,'/tmp/Simg2.nii.gz')

    '''
    

    '''    
    #create Nifti image
    S0img=ni.Nifti1Image(S0,np.eye(4))
    #transform volume
    S1img,A2=dp.volume_transform(S0img, A, reference=S0img,interp_order=3), A
    
    S1=S1img.get_data()
    S=np.zeros(S0.shape+(2,))
    S[:,:,:,0]=S0
    S[:,:,:,1]=S1
    
    #correct them
    S_corr,mats=corr.motion_correction(S,np.eye(4),ref=0,similarity='cr',subsampling=[1,1,1],interp='tri',order=3)

    print S.shape
    print S_corr.shape
    
    Simg=ni.Nifti1Image(S,np.eye(4))
    ni.save(Simg,'/tmp/Simg.nii.gz')   
    Simg2=ni.Nifti1Image(S_corr,np.eye(4))
    ni.save(Simg2,'/tmp/Simg2.nii.gz')

    #yield assert_array_almost_equal(xyz, pt)
    #return S,S_corr,mats

    '''
    

