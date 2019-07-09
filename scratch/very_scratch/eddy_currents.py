import numpy as np
import dipy as dp
import nibabel as ni

dname = '/home/eg01/Data_Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'
#dname =  '/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'


data,affine,bvals,gradients=dp.load_dcm_dir(dname)

"""
rot=np.array([[1,0,0,0],
              [0,np.cos(np.pi/2),-np.sin(np.pi/2),0],
              [0,np.sin(np.pi/2), np.cos(np.pi/2),0],
              [0,0,0,1]])

from scipy.ndimage import affine_transform as aff

naffine=np.dot(affine,rot)
"""

data[:,:,:,1]

source=ni.Nifti1Image(data[:,:,:,1],affine)
target=ni.Nifti1Image(data[:,:,:,0],affine)

#similarity 'cc', 'cr', 'crl1', 'mi', je', 'ce', 'nmi', 'smi'.  'cr'
similarity='cr'

#interp 'pv', 'tri'
interp =  'tri'

#subsampling None or sequence (3,)
subsampling=None

#search 'affine', 'rigid', 'similarity' or ['rigid','affine']
search='affine'

#optimizer 'simplex', 'powell', 'steepest', 'cg', 'bfgs' or
#sequence of optimizers
optimizer= 'powell'

T=dp.volume_register(source,target,similarity,\
                       interp,subsampling,search,)

sourceT=dp.volume_transform(source, T.inv(), reference=target)

s=source.get_data()
t=target.get_data()
sT=sourceT.get_data()
