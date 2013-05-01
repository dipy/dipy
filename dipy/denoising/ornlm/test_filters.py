import ornlm_module as ornlm
import numpy as np
import nibabel as nib
from hsm import hsm
from ascm import ascm
def test_filters():
    nib_image=nib.load("data/fibercup-averaged_b-1500.nii")
    image=nib_image.get_data().astype(np.double)
    affine=nib_image.get_affine()

    fima1=np.empty_like(image, order='F')
    fima2=np.empty_like(image, order='F')
    fima3=np.empty_like(image, order='F')
    fima4=np.empty_like(image, order='F')
    for i in xrange(image.shape[3]):
        print "Filtering volume",i+1,"/",image.shape[3]
        mv=image[:,:,:,i].max()
        fima1[:,:,:,i]=ornlm.ornlmpy(image[:,:,:,i], 3, 1, 0.05*mv)
        fima2[:,:,:,i]=ornlm.ornlmpy(image[:,:,:,i], 3, 2, 0.05*mv)
        fima4[:,:,:,i]=np.array(ascm(image[:,:,:,i], fima1[:,:,:,i],fima2[:,:,:,i], 0.05*mv))
        fima3[:,:,:,i]=np.array(hsm(fima1[:,:,:,i],fima2[:,:,:,i]))
    #####ornlm######
    nii_matlab_filtered1=nib.load('data/filtered_3_1_1.nii');
    matlab_filtered1=nii_matlab_filtered1.get_data().astype(np.double)
    diff1=abs(fima1-matlab_filtered1)
    print "Maximum error [ornlm (block size= 3x3)]: ", diff1.max()
    #####ornlm######
    nii_matlab_filtered2=nib.load('data/filtered_3_2_1.nii');
    matlab_filtered2=nii_matlab_filtered2.get_data().astype(np.double)
    diff2=abs(fima2-matlab_filtered2)
    print "Maximum error [ornlm (block size= 5x5)]: ", diff2.max()
    #######hsm########
    nii_matlab_filtered3=nib.load('data/filtered_hsm.nii');
    matlab_filtered3=nii_matlab_filtered3.get_data().astype(np.double)
    diff3=abs(fima3-matlab_filtered3)
    print "Maximum error [hsm]: ", diff3.max()
    #######ascm########
    nii_matlab_filtered4=nib.load('data/filtered_ascm.nii');
    matlab_filtered4=nii_matlab_filtered4.get_data().astype(np.double)
    diff4=abs(fima4-matlab_filtered4)
    print "Maximum error [ascm]: ", diff4.max()

test_filters()
