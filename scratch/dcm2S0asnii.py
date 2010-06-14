from dipy.io import dicomreaders as dcm
import nibabel as ni
import numpy as np
import dipy.core.generalized_q_sampling as gq


dname='/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'

S0name='/tmp/S0.nii'

smallname='/home/eg01/Devel/dipy/dipy/core/tests/data/small_volume2.5_steam_4000.nii'

smallname_grad = '/home/eg01/Devel/dipy/dipy/core/tests/data/small_volume2.5_steam_4000.gradients'

smallname_bvals = '/home/eg01/Devel/dipy/dipy/core/tests/data/small_volume2.5_steam_4000.bvals'

#read diffusion dicoms

data,affine,bvals,gradients=dcm.read_mosaic_dir(dname)

print data.shape

#calculate QA
gqs = gq.GeneralizedQSampling(data,bvals,gradients)

#gqs.QA[0]

S0 = data[:,:,:,0]



'''

#save the structural volume

img=ni.Nifti1Image(S0,affine)

ni.save(img,S0name)

#save the small roi volume

small= data[35:55,55:75,20:30,:]

imgsmall=ni.Nifti1Image(small,affine)

ni.save(img,smallname)

#save b-values and b-vecs

np.save(smallname_grad,gradients)

np.save(smallname_bvals,bvals)

'''
