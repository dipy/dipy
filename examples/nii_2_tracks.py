""" 
===============================================================================
Reconstruct Diffusion Data and Create Tractographies with Two Different Methods
===============================================================================

First import the necessary modules

numpy is for numerical computation
"""

import numpy as np
"""
nibabel is for data formats
"""
import nibabel as nib
"""
dipy.reconst is for reconstruction algorithms
"""
import dipy.reconst.gqi as gqi
import dipy.reconst.dti as dti
"""
dipy.tracking is for tractography algorithms
"""
from dipy.tracking.propagation import EuDX
"""
dipy.data is for using small test and example datasets
"""
from dipy.data import get_data

""" The datasets used need to have isotropic voxel size. If you have datasets with anisotropic voxel size 
then you can have a look at the example resample_aniso_2_iso.py

get_data is a simple function which brings here a few voxels to play with. In order to make this work with your data you should comment the 
line below and add the paths for your nifti file (*.nii or *.nii.gz) and your *.bvec and *.bval files. If you are not using nifti files or you don't
know how to create the *.bvec and *.bval files from your raw dicom (*.dcm) data then you can either try the example called dcm_2_tracks.py or use mricron
to convert the dicom files to nii, bvec and bval. 
"""
fimg,fbvals,fbvecs=get_data('small_101D')
""" Load the nifti file found at path fimg as an Nifti1Image 
"""
img=nib.load(fimg)
""" Read the datasets from the Nifti1Image
"""
data=img.get_data()
""" as you would expect data is 4-dimensional as you have one 3d volume for every gradient direction
"""
print('data.shape (%d,%d,%d,%d)' % data.shape)
""" Read the affine matrix which gives the mapping between volume indices and world coordinates
"""
affine=img.get_affine()
""" Read the b-values
"""
bvals=np.loadtxt(fbvals)
""" Read the b-vectors
"""
gradients=np.loadtxt(fbvecs).T
""" Calculate the single Tensor
"""
ten=dti.Tensor(data,bvals,gradients,thresh=50)
""" Calculate Fractional Anisotropy (FA) from the single tensor
"""
FA=ten.fa()
print('FA.shape (%d,%d,%d)' % FA.shape)
""" Generate tractography using Euler Delta Crossings (EuDX)

The basic input of EuDX is an anisotropic scalar metric and the integer indices for the peaks on the sampling sphere.
Other import options are the number of random seeds where the track will start propagate and the stopping criteria for
example     
"""
eu=EuDX(a=FA,ind=ten.ind(),seed_no=10000,a_low=.2)
""" 
EuDX returns a generator class which yields a track each time this class is called. 
With that way we can generate millions of tracks without using any substantial amount of memory. 
For an example of what to do when you want to generate millions of tracks have a look at save_dpy.py

However, in this example that we only have 10000 seeds we have loaded all tracks in a list using list comprehension without worry much about memory.  
"""
ten_tracks=[track for track in eu]
""" In dipy we usually represent tractography as a list of tracks where every track is a numpy array of shape (N,3) where N is the number of points of the track. 
"""
print('Number of FA tracks %d' % len(ten_tracks))
""" Another way to represent tractography is as a numpy array of numpy objects. This way has an additional advantage that it can be saved very easily using the numpy. In theory in a list is faster to append an element and in an array is faster to access. In other words both representations have different + and -. Other representations are possible too e.g. graphtheoretic.
"""
ten_tracks_asobj=np.array(ten_tracks,dtype=np.object)
np.save('ten_tracks.npy',ten_tracks_asobj)
print('FA tracks saved in ten_tracks.npy')
""" You probably have heard about the problem of crossings in diffusion MRI. 
The single tensor model cannot detect a crossing.
With Generalized Q-Sampling this is possible even up to a quadruple crossing 
or even higher as long as your datasets are able to provide that resolution.
"""
gqs=gqi.GeneralizedQSampling(data,bvals,gradients)
QA=gqs.qa()
print('QA.shape (%d,%d,%d,%d)' % QA.shape)
""" The outcome of gqs is Quantitative Anisotropy a metric much different than FA even in shape however this can be feeded again at the
same EuDX class with no problem. This is one of the advantages of EuDX that it can be used with all know model-based methods for example Single Tensor,Multiple Tensor,Stick & Ball, Higher Order Tensor and model-free methods DSI, QBall, GQI etc.
We created this so we can compare tractographies generated from very different models.  
"""
eu2=EuDX(a=QA,ind=gqs.ind(),seed_no=10000,a_low=.0239)
gqs_tracks=[track for track in eu2]
print('Number of QA tracks %d' % len(gqs_tracks))
""" Do you see the difference between the number of gqs tracks and ten tracks? Can you think a reason why? Correct, CROSSINGS!!!
"""
gqs_tracks_asobj=np.array(gqs_tracks,dtype=np.object)
np.save('gqs_tracks.npy',ten_tracks_asobj)
print('QA tracks saved in gqs_tracks.npy')
"""
This is the end of this very simple example you can load again the saved tracks using np.load
"""
from dipy.viz import fvtk
r=fvtk.ren()
fvtk.add(r,fvtk.line(ten_tracks,fvtk.red,opacity=0.1))
gqs_tracks2=[t+np.array([10,0,0]) for t in gqs_tracks]
fvtk.add(r,fvtk.line(gqs_tracks2,fvtk.green,opacity=0.1))
fvtk.show(r)

''' 
T2=tp.FACT_Delta(ten.FA,ten.IN,seeds_no=10000,qa_thr=0.2).tracks
t6=time.clock()
print ('Create %d FA tracks in %d secs' %(len(T2),t6-t5))

T2=[t+np.array([100,0,0]) for t in T2]

print('Red tracks propagated based on QA')
print('Green tracks propagated based on FA')
r=fos.ren()
fos.add(r,fos.line(T,fos.red))
fos.add(r,fos.line(T2,fos.green))
fos.show(r)
'''



