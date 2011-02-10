
""" 

====================
Visualize Crossings
====================

Overview
========

**This example visualizes the crossings structure of a few voxels.**

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

If you are not using nifti files or you don't know how to create the ``*.bvec``
and ``*.bval`` files from your raw dicom (``*.dcm``) data then you can either
try the example called ``dcm_2_tracks.py`` or use mricron_ to convert the dicom
files to nii, bvec and bval files using ``dcm2nii``. 
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
**Read the b-values** which are a function of the strength, duration, temporal spacing and timing parameters of the 
specific paradigm used in the scanner, one per gradient direction.
"""

bvals=np.loadtxt(fbvals)

""" 
**Read the b-vectors**, the unit gradient directions.
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
  
The QA array is 
significantly different in shape from the FA array, 
however it too can be directly input to the EuDX class:

We explore the voxel [0,0,0].
"""

qa=QA[0,0,0]

"""
``qa`` is the quantitative anisotropy metric
"""

IN=gqs.ind()
ind=IN[0,0,0]

"""
``ind`` holds the indices of the vertices of (up to 5) gqi odf local maxima
"""

print 'quantitative anisotropy metric =', qa
print 'indices of local gqi odf maxima =', ind

"""
There are approximately equal maxima in the directions of vertices 117 and 1. To find out
where these are we need to work with the symmetric 362 vertex sphere on which 
the reconstruction was performed. 
"""

from dipy.data import get_sphere
verts, faces = get_sphere('symmetric362')

from dipy.viz import fvtk

r=fvtk.ren()

print 'Vertex 117 is', verts[117]
print 'Vertex 1 is', verts[1]
print 'The number of local maxima is', np.sum(ind>0)

"""
- Vertex 117 is [ 0.54813892  0.76257497  0.34354511]
- Vertex 1 is [ 0.0566983   0.17449942  0.98302352]
- The number of local maxima is 2
"""

summary = []
for i, index in enumerate(np.ndindex(QA.shape[:3])):
    if QA[index][0] > .0239:
        summary.append([index, np.sum(IN[index]>0), QA[index]])
        #print i, index, np.sum(IN[index]>0), QA[index]

print "There are %d suprathreshold voxels" % len(summary)
maxcounts = np.zeros(10,'int')
for voxel, count, indices in summary:
    maxcounts[count]+=1
#print maxcounts[maxcounts>0]

"""
We are using a fairly low threshold of 0.0239 and all 600 voxels are suprathreshold.

maxcounts[maxcounts>0] = [  0 405 152  30  10], so there are 

- 405 voxels with a single maximum (no crossing), 
- 152 with 2 maxima, 
- 30 voxels with 3 maxima, 
- 10 voxels with 4 maxima, 
- and 3 voxels with (at least) 5 maxima.  

We locate 3 contiguous voxels [3,8,4], [3,8,5], and [3,8,6] which have respectively
1, 2, and 3 crossings.

``fvtk.crossing`` is a helper function which we use to graph the orientations of the maxima 
of all the voxels in our dataset. We use 3 different colourings and offset the graphs to display them 
in one diagram. The colourings are:

- all blue, with the 3 voxels used above ([3,8,4], [3,8,5], and [3,8,6]) marked in blue, indigo, and red.
- the Boys' colour map (see ``colormap.boys2rgb.py``)
- the orientation colour map (see ``colormap.orient2rgb.py`` with red: left-right; green: anteroposterior; blue: superior-inferior. 

"""

#3,8,4 no crossing
no_cross=fvtk.crossing(QA[3,8,4],IN[3,8,4],verts,1)

#3,8,5 crossing
cross=fvtk.crossing(QA[3,8,5],IN[3,8,5],verts,1)

#3,8,6 double crossing
dcross=fvtk.crossing(QA[3,8,6],IN[3,8,6],verts,1)

all,allo=fvtk.crossing(QA,IN,verts,1,True)
fvtk.add(r,fvtk.line(all,fvtk.azure,linewidth=1.))

no_cross_shift=[c+np.array([3,8,4]) for c in no_cross]
cross_shift=[c+np.array([3,8,5]) for c in cross]
dcross_shift=[c+np.array([3,8,6]) for c in dcross]

fvtk.add(r,fvtk.line(no_cross_shift,fvtk.blue,linewidth=5.))
fvtk.add(r,fvtk.line(cross_shift,fvtk.indigo,linewidth=5.))
fvtk.add(r,fvtk.line(dcross_shift,fvtk.red,linewidth=5.))

from dipy.viz import colormap as cm

all_shift=[c+np.array([10,0,0]) for c in all]
all_shift2=[c+np.array([20,0,0]) for c in all]

colors=np.zeros((len(all),3))
colors2=np.zeros((len(all),3))
for (i,a) in enumerate(all):
    #print a[0]
    colors[i]=cm.boys2rgb(allo[i])
    colors2[i]=cm.orient2rgb(allo[i])

fvtk.add(r,fvtk.line(all_shift,colors,linewidth=1.))
fvtk.add(r,fvtk.line(all_shift2,colors2,linewidth=2.))


"""

.. figure:: visualize_cross1000000.png
   :align: center

   **The crossings of a region of interest shown with one color, or boy2rgb or standard orient2rgb colormap**.

"""

# To show the figure
# fvtk.show(r,size=(800,800))

# Here's how we make the illustration.
print('Saving illustration as visualize_cross1000000.png')
fvtk.record(r,n_frames=1,out_path='visualize_cross',size=(600,600))
