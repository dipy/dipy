''' Anisotropic to isotropic voxel conversion 
'''
import numpy as np

def resampl_aniso2iso(faniso,faniso2):
    '''  resample data from anisotropic to isotropic voxel size
    '''
    pass

from dipy.data import get_data

fimg,fbvals,fbvecs=get_data('small_101D')
bvals=np.loadtxt(fbvals)
bvecs=np.loadtxt(fbvecs).T

import nibabel as nib
img=nib.load(fimg)
data=img.get_data()
affine=img.get_affine()
print affine
print img.get_header().get_zooms()

U,s,Vh = np.linalg.svd(affine[:3,:3],full_matrices=False)
S=np.diag(s)

np.set_printoptions(2)
#print np.dot(evecs,np.dot(np.diag(evals),np.linalg.inv(evecs)))

print np.dot(U,np.dot(S,Vh))
print S

S2=np.diag([4,4,4])
print S2
R=np.dot(U,np.dot(S2,Vh))
affine2=affine.copy()
affine2[:3,:3]=R
from scipy.ndimage import affine_transform

print affine[:3,3]
print data.shape
data2=affine_transform(data[...,0],R,affine[:3,3])

#print data.shape
print data2.shape






