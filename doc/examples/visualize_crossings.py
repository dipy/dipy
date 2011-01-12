
""" 
.. include:: ../links_names.txt

====================
Visualize Crossings
====================

Overview
========

**This example visualizes a few voxels.**

First import the necessary modules
----------------------------------

* ``numpy`` is for numerical computation

"""

import numpy as np

"""
* ``nibabel`` is for data formats
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
* ``dipy.data`` is for small datasets we use in tests and examples.
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
from your raw dicom (``*.dcm``) data then you can either try the example called ``dcm_2_tracks.py`` or use _mricron
to convert the dicom files to nii, bvec and bval files using ``dcm2nii``. 
"""

fimg,fbvals,fbvecs=get_data('small_101D')

""" 
* **Load the nifti file found at path fimg as an Nifti1Image.**
"""

img=nib.load(fimg)

""" 
* **Read the datasets from the Nifti1Image.**
"""

data=img.get_data()
print('data.shape (%d,%d,%d,%d)' % data.shape)

""" 
This produces the output::

  data.shape (6,10,10,102)

As you would expect, the raw diffusion weighted MR data is 4-dimensional as 
we have one 3-d volume (6 by 10 by 10) for each gradient direction.

* **Read the affine matrix**
  which gives the mapping between volume indices (voxel coordinates) and world coordinates.
"""

affine=img.get_affine()

""" 
* **Read the b-values** which are a function of the strength, duration, temporal spacing and timing parameters of the 
  specific paradigm used in the scanner, one per gradient direction.
"""

bvals=np.loadtxt(fbvals)

""" 
* **Read the b-vectors**, the unit gradient directions.
"""

gradients=np.loadtxt(fbvecs).T

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

"""
**This is the end of this very simple example** You can reload the saved tracks using 
``np.load`` from your current directory. You can optionaly install ``python-vtk``
and visualize the tracks using ``fvtk``:
"""


qa=QA[0,0,0]
IN=gqs.ind()
ind=IN[0,0,0]
print qa
print ind

from dipy.data import get_sphere

fname=get_sphere('symmetric362')
sph=np.load(fname)
verts=sph['vertices']
faces=sph['faces']

from dipy.viz import fvtk

r=fvtk.ren()


print verts[117]
print verts[1]
print np.sum(ind>0) 

for index in np.ndindex(QA.shape[:3]):
    if QA[index][0] > .0239:
        print index, np.sum(IN[index]>0), QA[index]

#print QA[3,8,6]
#print IN[3,8,6]

def cross(qa,ind,verts,scale=1):
    Ts=[]
    print qa
    print ind    
    for (i,_i) in enumerate(ind):
        if _i > 0:
            Ts.append([scale*qa[i]*np.vstack((verts[_i],-verts[_i]))])
    return Ts

Ts2=cross(QA[3,8,4],IN[3,8,4],verts,scale=0.3)
for T in Ts2:
    T[0]=T[0]+np.array([-.2,0,0])
    fvtk.add(r,fvtk.line(T,fvtk.indigo,linewidth=10.))
#fvtk.show(r)

Ts3=cross(QA[3,8,5],IN[3,8,5],verts)
for T in Ts3:    
    T[0]=T[0]+np.array([-.1,0,0])
    fvtk.add(r,fvtk.line(T,fvtk.blue,linewidth=10.))
#fvtk.show(r)

Ts=cross(QA[3,8,6],IN[3,8,6],verts)
for T in Ts:    
    fvtk.add(r,fvtk.line(T,fvtk.azure,linewidth=10.))
fvtk.show(r,png_magnify=1)


        
    
"""
**Hope that helps!**
---------------------
"""

