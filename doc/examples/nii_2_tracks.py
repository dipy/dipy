
""" 
.. include:: ../links_names.txt

=============================
From niftis to tractographies
=============================

Overview
========

**This example gives a tour of many of the features of ``dipy``.**

First import the necessary modules
==================================

* ``numpy`` is for numerical computation

"""

import numpy as np

"""
* ``nibabel`` is for data formats**
"""

import nibabel as nib

"""
* ``dipy.reconst`` is for the reconstruction algorithms which we use to create directionality models 
for a voxel from the raw data. 
"""

import dipy.reconst.gqi as gqi
import dipy.reconst.dti as dti

"""
* ``dipy.tracking`` is for tractography algorithms which create sets of tracks by integrating 
directionality models across voxels.
"""

from dipy.tracking.propagation import EuDX

"""
* ``dipy.data`` is for small datasets for use in tests and examples.
"""

from dipy.data import get_data


""" 
Isotropic voxel sizes required
==============================
``dipy`` requires its datasets to have isotropic voxel size. If you have datasets with anisotropic voxel size 
then you need to resample with isotropic voxel size. We have provided an algorithm for this. 
You can have a look at the example ``resample_aniso_2_iso.py``

Accessing the necessary datasets
================================
``get_data`` is provides data for a small region of interest from a real
diffusion weighted MR dataset acquired with 101 gradient. 
In order to make this work with your data you should comment out the line below and add the paths 
for your nifti file (``*.nii`` or ``*.nii.gz``) and your ``*.bvec`` and ``*.bval files``. If you are not using 
nifti files or you don't know how to create the ``*.bvec`` and ``*.bval`` files from your raw dicom (``*.dcm``) 
data then you can either try the example called ``dcm_2_tracks.py`` or use _mricron
to convert the dicom files to nii, bvec and bval files using ``dcm2nii``. 
"""

fimg,fbvals,fbvecs=get_data('small_101D')

""" 
* **Load the nifti file found at path ``fimg`` as an Nifti1Image.**
"""

img=nib.load(fimg)

""" 
* **Read the datasets from the Nifti1Image.**
"""

data=img.get_data()
print('data.shape (%d,%d,%d,%d)' % data.shape)

""" 
As you would expect, your raw diffusion weighted MR data is 4-dimensional as 
you have one 3d volume for each gradient direction.

* **Read the affine matrix**
which gives the mapping between volume indices (voxel coordinates) and world coordinates.
"""

affine=img.get_affine()

""" 
* **Read the b-values**
these are a function of the strength, duration, temporal spacing and timing parameters of the 
specific paradigm used in the scanner, one per gradient direction.
"""

bvals=np.loadtxt(fbvals)

""" 
* **Read the b-vectors - unit gradient directions.**
"""

gradients=np.loadtxt(fbvecs).T

""" 
Calculating models and parameters of directionality
---------------------------------------------------
We are now set up with all the data and parameters to start calculating directional models 
for voxels and their associated parameters, e.g. anisotropy.

* **Calculate the single tensor model.**  
"""

ten=dti.Tensor(data,bvals,gradients,thresh=50)

""" 
* **Calculate Fractional Anisotropy (FA) from the single tensor model**
"""

FA=ten.fa()
print('FA.shape (%d,%d,%d)' % FA.shape)

""" 
Generate a tractography 
-----------------------
Here we use the Euler Delta Crossings (EuDX) alggorithm.
The main parameters of ``EuDX`` are 
  - an anisotropic scalar metric e.g. FA, and 
  - the indices for the peaks on the sampling sphere. 
Other important options are 
  - the number of random seeds where the track propagation is initiated, and 
  - the stopping criteria, for example a low threshold for anisotropy. For instance 
if we are using fractional anisotropy (FA) a typical threshold value might be ``a_low=.2``    
"""

eu=EuDX(a=FA,ind=ten.ind(),seed_no=10000,a_low=.2)

""" 
EuDX returns a generator class which yields a further track each time this class is called. 
In this way we can generate millions of tracks without using a substantial amount of memory. 
For an example of what to do when you want to generate millions of tracks have a look at 
``save_dpy.py`` in the ``examples`` directory.

However, in the current example that we only have 10000 seeds, we have loaded all tracks in a list 
using list comprehension([]) without worry much about memory.  
"""

ten_tracks=[track for track in eu]

""" 
In dipy we usually represent tractography as a list of tracks where every track is a numpy array of shape (N,3) where N is the number of points of each track. 
"""

print('Number of FA tracks %d' % len(ten_tracks))

""" 
Another way to represent tractography is as a numpy array of numpy objects. 
This way has an additional advantage that it can be saved very easily using the numpy. 
In theory in a list is faster to append an element and in an array is faster to access. 
In other words both representations have different + and -. 
Other representations are possible too e.g. graphtheoretic etc.
"""

ten_tracks_asobj=np.array(ten_tracks,dtype=np.object)
np.save('ten_tracks.npy',ten_tracks_asobj)
print('FA tracks saved in ten_tracks.npy')

""" 
Crossings and GQS
-----------------
You probably have heard about the problem of crossings in diffusion MRI. a metric rather
The single tensor model cannot detect a simple crossing of two fibres.
With Generalized Q-Sampling (GQS) this is possible even up to a quadruple crossing 
or even higher so long as your datasets are able to provide that resolution.
"""

gqs=gqi.GeneralizedQSampling(data,bvals,gradients)
QA=gqs.qa()
print('QA.shape (%d,%d,%d,%d)' % QA.shape)

""" 
A useful metric derived from GQS is *Quantitative Anisotropy* (QA). The QA function on the sphere is 
significantly different in shape from FA, however it too can be directly input to the EuDX class.
 
This is one of the advantages of EuDX that it can be used with a wide range of model-based methods, such as 
  - Single Tensor,
  - Multiple Tensor, 
  - Stick & Ball, 
  - Higher Order Tensor, and 
  - model-free methods such as DSI, QBall, GQI etc.

We designed this algorithm so we that we can compare directly tractographies generated 
from very different models or choices of threshold.  
"""

eu2=EuDX(a=QA,ind=gqs.ind(),seed_no=10000,a_low=.0239)
gqs_tracks=[track for track in eu2]
print('Number of QA tracks %d' % len(gqs_tracks))

""" 
Do you see the difference between the number of gqs_tracks and ten_tracks? Can you think of a reason? 
Correct, *crossings*! When the underlying directionality model supports crossings then 
distinct tracks will be propagated from a seed towards the different directions in equal abundance.
  
In ``dipy`` it is very easy to count the number of crossings in a voxel, volume or region of interest

"""

gqs_tracks_asobj=np.array(gqs_tracks,dtype=np.object)
np.save('gqs_tracks.npy',gqs_tracks_asobj)
print('QA tracks saved in gqs_tracks.npy')

"""
 **This is the end of this very simple example you can load again the saved tracks using ``np.load`` from your current directory. You can optionaly install python-vtk
and visualize the tracks using ``fvtk``.**
"""

from dipy.viz import fvtk
r=fvtk.ren()
fvtk.add(r,fvtk.line(ten_tracks,fvtk.red,opacity=0.1))
gqs_tracks2=[t+np.array([10,0,0]) for t in gqs_tracks]
fvtk.add(r,fvtk.line(gqs_tracks2,fvtk.green,opacity=0.1))
#fvtk.show(r)

"""
**Thank you!**
--------------
"""

