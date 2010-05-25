from dipy.io import dicomreaders as dcm
import nibabel as ni
import numpy as np
from dipy.core import stensor as sten


dname='/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'

S0name='/tmp/S0.nii'


data,affine,bvals,gradients=dcm.read_mosaic_dir(dname)

print data.shape

S0 = data[:,:,:,0]

img=ni.Nifti1Image(S0,affine)

ni.save(img,S0name)

