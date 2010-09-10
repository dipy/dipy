import numpy as np
import dipy as dp
import nibabel as ni
from scipy.ndimage import rotate
from nipy.neurospin.registration import resample, IconicRegistration, Rigid
from nipy import load_image, save_image

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
    
    #S0=rotate(S0,5,reshape=False)
    
    S1=rotate(S0,30,reshape=False)
    S=np.zeros(S0.shape+(2,))
 
    S0img=ni.Nifti1Image(S0,np.eye(4))
    ni.save(S0img,'/tmp/S0img.nii.gz')

    S1img=ni.Nifti1Image(S1,np.eye(4))
    ni.save(S1img,'/tmp/S1img.nii.gz')

    S0img=load_image('/tmp/S0img.nii.gz')
    S1img=load_image('/tmp/S1img.nii.gz')

    save_image(S0img,'/tmp/S0img.nii.gz')
    save_image(S1img,'/tmp/S1img.nii.gz')
    
    #T=register(S1img,S0img,interp='pv')
    #NS1img=transform(S1img, T)
    #ni.save(NS1img,'/tmp/NS1img.nii.gz')    

    R = IconicRegistration(S1img, S0img)
    
    # To speed up computation (set subsampling factors so that the
    # number of voxels considered for registration is roughly 40**3)
    R.set_source_fov(fixed_npoints=40**3)

    #Tell the registration algorithm to look for a rigid transform (6 parameters)
    T = Rigid()

    # Apply a strong initial rotation of 1 rad around z-axis at the image origin
    # (as implicitly defined by the 'affine', i.e. top left corner in your case)
    # The following syntax is obviously to be improved...
    T.param = [0,0,0,0,0,1.]/T.precond[0:6]

    #Run registration
    T2=R.optimize(T)

    
    print 'T2',T2

    #Finally, resample the ***target*** image using T (or the source using T.inv())
    S0img_resampled = resample(S0img, T2)
    
    save_image(S0img_resampled,'/tmp/NS1img.nii.gz')

    

