import numpy as np

import nibabel as ni
from nibabel.dicom import dicomreaders as dcm

import dipy.core.generalized_q_sampling as gq


dname='/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'

#dname ='/home/eg309/Data/Eleftherios/Series_003_CBU_DTI_64D_iso_1000'

S0name='/tmp/S0.nii'

#smallname='/tmp/small_volume2.5_steam_4000.nii'

smallname='/tmp/small_64D.nii'

smallname_grad = '/tmp/small_64D.gradients'

smallname_bvals = '/tmp/small_64D.bvals'


#read diffusion dicoms

data,affine,bvals,gradients=dcm.read_mosaic_dir(dname)

print data.shape

#calculate QA
#gqs = gq.GeneralizedQSampling(data,bvals,gradients)

#gqs.QA[0]

#S0 = data[:,:,:,0]

"""

#save the structural volume

#img=ni.Nifti1Image(S0,affine)

#ni.save(img,S0name)

#save the small roi volume

#small= data[35:55,55:75,20:30,:]


small= data[54:64,54:64,30:40,:]

naffine = np.dot(affine, np.array([[1,0,0,54],[0,1,0,54],[0,0,1,30],[0,0,0,1]]))

imgsmall=ni.Nifti1Image(small,naffine)

ni.save(imgsmall,smallname)

#save b-values and b-vecs

np.save(smallname_grad,gradients)

np.save(smallname_bvals,bvals)

"""
