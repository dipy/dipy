""" 

===============================
From raw data to tractographies
===============================

Overview
========

**This example gives a tour of some simple features of dipy.**

First import the necessary modules
----------------------------------

``numpy`` is for numerical computation

"""

import numpy as np

"""
``nibabel`` is for data formats
"""

import nibabel as nib

"""
``dipy.reconst`` is for the reconstruction algorithms which we use to create directionality models 
for a voxel from the raw data. 
"""

import dipy.reconst.gqi as gqi
import dipy.reconst.dti as dti

"""
``dipy.tracking`` is for tractography algorithms which create sets of tracks by integrating 
  directionality models across voxels.
"""

from dipy.tracking.propagation import EuDX

"""
``dipy.data`` is for small datasets we use in tests and examples.
"""

from dipy.data import get_data


""" 
Isotropic voxel sizes required
------------------------------
``dipy`` requires its datasets to have isotropic voxel size. If you have datasets with anisotropic voxel size 
then you need to resample with isotropic voxel size. We have provided an algorithm for this. 
You can have a look at the example ``resample_aniso_2_iso.py``

Accessing the necessary datasets
--------------------------------
``get_data`` provides data for a small region of interest from a real
diffusion weighted MR dataset acquired with 102 gradients (including one for b=0). 

In order to make this work with your data you should comment out the line below and add the paths 
for your nifti file (``*.nii`` or ``*.nii.gz``) and your ``*.bvec`` and ``*.bval files``. 

If you are not using nifti files or you don't know how to create the ``*.bvec`` and ``*.bval`` files 
from your raw dicom (``*.dcm``) data then you can either try recent module nibabel.nicom 
"""

try:
    from nibabel.nicom.dicomreaders import read_mosaic_dir
except:
    print('nicom for dicom is not installed')

"""
or to convert the dicom files to nii, bvec and bval files using ``dcm2nii``. 
"""

fimg,fbvals,fbvecs=get_data('small_101D')

""" 
**Load the nifti file found at path fimg as an Nifti1Image.**
"""

img=nib.load(fimg)

""" 
**Read the datasets from the Nifti1Image.**
"""

data=img.get_data()
print('data.shape (%d,%d,%d,%d)' % data.shape)

""" 
This produces the output::

  data.shape (6,10,10,102)

As you would expect, the raw diffusion weighted MR data is 4-dimensional as 
we have one 3-d volume (6 by 10 by 10) for each gradient direction.

**Read the affine matrix**
  which gives the mapping between volume indices (voxel coordinates) and world coordinates.
"""

affine=img.get_affine()

""" 
**Read the b-values** which are a function of the strength, duration, temporal spacing 
and timing parameters of the specific paradigm used in the scanner, one per gradient direction.
"""

bvals=np.loadtxt(fbvals)

""" 
**Read the b-vectors**, the unit gradient directions.
"""

gradients=np.loadtxt(fbvecs).T

""" 
Calculating models and parameters of directionality
---------------------------------------------------
We are now set up with all the data and parameters to start calculating directional models 
for voxels and their associated parameters, e.g. anisotropy.

**Calculate the Single Tensor Model (STM).**  
"""

ten=dti.Tensor(data,bvals,gradients,thresh=50)

""" 
**Calculate Fractional Anisotropy (FA) from STM**
"""

FA=ten.fa()
print('FA.shape (%d,%d,%d)' % FA.shape)

"""
As expected the FA is a 3-d array with one value per voxel::

  FA.shape (6,10,10)
  
Generate a tractography 
-----------------------
Here we use the Euler Delta Crossings (EuDX) algorithm.
The main input parameters of ``EuDX`` are 

  * an anisotropic scalar metric e.g. FA
  * the indices for the peaks on the sampling sphere. 
  
Other important options are 

  * the number of random seeds where the track propagation is initiated, 
  * a stopping criterion, for example a low threshold for anisotropy. For instance 
    if we are using *Fractional Anisotropy (FA)* a typical threshold value might be ``a_low=.2``
    
"""

eu=EuDX(a=FA,ind=ten.ind(),seeds=10000,a_low=.2)

""" 
EuDX returns a generator class which yields a further track each time this class is called. 
In this way we can generate millions of tracks without using a substantial amount of memory. 
For an example of what to do when you want to generate millions of tracks with minimum memory usage have a look at 
``save_dpy.py`` in the ``examples`` directory. However, in the current example that we only have 10000 seeds, and we can load all tracks 
in a list using list comprehension([]) without having to worry about memory.  
"""

ten_tracks=[track for track in eu]

""" 
In dipy we usually represent tractography as a list of tracks. Every track is a numpy array of shape (N,3) 
where N is the number of points in the track. 
"""

print ('The number of FA tracks is %d' % len(ten_tracks))
print ('The number of points in ten_tracks[130] is %d' % len(ten_tracks[130]))
print ('The points in ten_tracks[130] are:')
print ten_tracks[130]

"""
As we use random seeding for the tractography the results will differ when repeated, however
one run gave us the following information::

  The number of FA tracks is 8280
  The number of points in ten_track[130] is 7
  The points in ten_tracks[130] are:
  [[ 1.73680878  5.08249903  4.48492956]
   [ 1.45797026  4.76981783  4.21201992]
   [ 1.14244306  4.46308756  3.97461915]
   [ 0.84001541  4.14648438  3.73316503]
   [ 0.53758776  3.82988143  3.49171114]
   [ 0.22055824  3.52935386  3.24845099]
   [ 0.22055824  3.52935386  3.24845099]]

Another way to represent tractography is as a numpy array of numpy objects. 
This way has an additional advantage that it can be saved very easily using numpy utilities. 
In theory, in a list it is faster to append an element, and in an array is faster to access. 
In other words both representations have different pros and cons. 
Other representations are possible too e.g. graphtheoretic etc.
"""

ten_tracks_asobj=np.array(ten_tracks,dtype=np.object)
np.save('ten_tracks.npy',ten_tracks_asobj)
print('FA tracks saved in ten_tracks.npy')

""" 
Crossings and Generalized Q-Sampling
------------------------------------
You probably have heard about the problem of crossings in diffusion MRI. 
The single tensor model cannot detect a simple crossing of two fibres. 
However with *Generalized Q-Sampling (GQS)* this is possible even up to a quadruple crossing 
or higher depending on the resolution of your datasets. Resolution will 
typically depend on signal-to-noise ratio and voxel-size.
"""

gqs=gqi.GeneralizedQSampling(data,bvals,gradients)

"""
A useful metric derived from GQS is *Quantitative Anisotropy* (QA). 
"""

QA=gqs.qa()
print('QA.shape (%d,%d,%d,%d)' % QA.shape)

"""
QA is a 4-d array with up to 5 peak QA values for each voxel::

  QA.shape (6,10,10,5)
  
QA array is 
significantly different in shape from the FA array, 
however it too can be directly input to the EuDX class:
"""

eu2=EuDX(a=QA,ind=gqs.ind(),seeds=10000,a_low=.0239)

"""
This shows one of the advantages of our EuDX algorithm: it can be used with a wide range of model-based methods, such as 
  - Single Tensor
  - Multiple Tensor 
  - Stick & Ball
  - Higher Order Tensor 

and model-free methods such as 
  - DSI
  - QBall
  - GQI *etc.*

We designed the algorithm this way so we that we can compare directly tractographies generated
from the same dataset 
with very different models and/or choices of threshold.  

Now we look at the QA tractography:
"""

gqs_tracks=[track for track in eu2]
print('The number of QA tracks is %d' % len(gqs_tracks))

""" 
with output::

  The number of QA tracks is 14022

Note the difference between the number of gqs_tracks and ten_tracks. There are more with
QA than with FA. This is because of the 
presence of crossings which GQI can detect but STM cannot. When the underlying directionality model supports crossings then 
distinct tracks will be propagated from a seed towards the different directions in equal abundance.
  
In ``dipy`` it is very easy to count the number of crossings in a voxel, volume or region of interest

"""

gqs_tracks_asobj=np.array(gqs_tracks,dtype=np.object)
np.save('gqs_tracks.npy',gqs_tracks_asobj)
print('QA tracks saved in gqs_tracks.npy')

"""
**This is the end of this very simple example** You can reload the saved tracks using 
``np.load`` from your current directory. You can optionaly install ``python-vtk``
and visualize the tracks using ``fvtk``:
"""

from dipy.viz import fvtk
r=fvtk.ren()
fvtk.add(r,fvtk.line(ten_tracks,fvtk.red,opacity=0.05))
gqs_tracks2=[t+np.array([10,0,0]) for t in gqs_tracks]
fvtk.add(r,fvtk.line(gqs_tracks2,fvtk.green,opacity=0.05))

"""
Press 's' to save this screenshot when you have displayed it with ``fvtk.show``.
Or you can even record a video using ``fvtk.record``.

You would show the figure with something like::

    fvtk.show(r,png_magnify=1,size=(600,600))

To record a video of 50 frames of png, something like::

    fvtk.record(r,cam_pos=(0,40,-40),cam_focal=(5,0,0),n_frames=50,magnification=1,out_path='nii_2_tracks',size=(600,600),bgr_color=(0,0,0))

.. figure:: nii_2_tracks1000000.png
   :align: center

   **Same region of interest with different underlying voxel representations generates different tractographies**.

"""

# Here's how we make the figure.
print('Saving illustration as nii_2_tracks1000000.png')
fvtk.record(r,n_frames=1,out_path='nii_2_tracks',size=(600,600))

