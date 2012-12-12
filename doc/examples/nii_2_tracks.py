""" 

==========================
Lets make some streamlines
==========================

Overview
--------

We will show how to generete streamlines with Dipy. First we will reconstruct
the datasets using a single Tensor model and next we will use a Constant Solid
Angle (QBall) model. EuDX will be used for the fiber tracking bits.

First import the necessary modules
----------------------------------

``numpy`` is for numerical computation

"""

import numpy as np

"""
``nibabel`` is for loading imaging datasets
"""

import nibabel as nib

"""
``dipy.reconst`` is for the reconstruction algorithms which we use to create voxel models from the raw data. 
"""

from dipy.reconst.dti import TensorModel

"""
``dipy.tracking`` is for tractography algorithms which create sets of tracks by integrating directionality models across voxels.
"""

from dipy.tracking.eudx import EuDX

"""
``dipy.data`` is used for small datasets that we use in tests and examples.
"""

from dipy.data import fetch_beijing_dti, read_beijing_dti

"""
Fetch will download the raw dMRI dataset of a single subject. The size of the dataset is 51 MBytes.
You only need to fetch once.
"""

fetch_beijing_dti()

"""
Next, we read the saved dataset
"""

img, gtab = read_beijing_dti()

data = img.get_data()
print('data.shape (%d,%d,%d,%d)' % data.shape)

"""
This dataset has anisotropic voxel sizes, therefore reslicing is necessary
"""

affine = img.get_affine()

"""
Load and show the zooms which hold the voxel size.
"""

zooms = img.get_header().get_zooms()[:3]
zooms

"""
``(1.79, 1.79, 2.5)``

Set the required new voxel size.
"""

new_zooms = (2., 2., 2.)
new_zooms

"""
``(2.0, 2.0, 2.0)``

Start reslicing. Trilinear interpolation is used by default.
"""

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine, zooms, new_zooms)
data2.shape

"""
Initiate your Model
"""

tenmodel = TensorModel(gtab)

mask = data2[..., 0] > 50

"""
Fit your data
"""

tenfit = tenmodel.fit(data2, mask)

FA = tenfit.fa

"""
Remove nans
"""

FA[np.isnan(FA)] = 0

"""
EuDX takes as input discretized directions on a unit sphere. Therefore,
it is necessary to discretize the eigen vectors before feeding them in EuDX.
"""

from dipy.reconst.dti import quantize_evecs

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

peak_indices = quantize_evecs(tenfit.evecs, sphere.vertices)

"""
EuDX is the fiber tracking algorithm that we use in this example.
The most important parameters is the first one which represents the
magnitude of the peak of a scalar anisotropic function, the 
second which represents the indices of the discretized directions of 
the peaks and odf_vertices are the vertices of the input sphere.
"""

from dipy.tracking.eudx import EuDX

eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices, a_low=0.2)

tensor_streamlines = [streamline for streamline in eu]

"""
We use here a python vtk module to show the streamlines
"""

from dipy.viz import fvtk

"""
Create a scene
"""

r=fvtk.ren()

"""
Every streamline will be coloured according to its orientation
"""

from dipy.viz.colormap import line_colors

"""
fvtk.line adds a streamline actor for streamline visualization
and fvtk.add adds this actor in the scene
"""

fvtk.add(r, fvtk.line(tensor_streamlines, line_colors(tensor_streamlines)))

print('Saving illustration as tensor_tracks.png')
fvtk.record(r, n_frames=1, out_path='tensor_tracking.png', size=(600, 600))

"""
Lets now repeat the same procedure with a CSA model
"""

from dipy.reconst.shm import CsaOdfModel, normalize_data

data2 = data2.astype('f8')

#normalize_data(data2, gtab.bvals, 1., out=data2)

"""
We instantiate our CSA model with sperical harmonic order of 4
"""

csamodel = CsaOdfModel(gtab, 4)

"""
Peaks from model is used to calculate properties of the ODFs and return for
example the peaks and their indices, or GFA which is similar to FA but for ODF
based models.
"""

from dipy.reconst.odf import peaks_from_model

peaks = peaks_from_model(model=csamodel, 
                         data=data2, 
                         sphere=sphere, 
                         relative_peak_threshold=.8,
                         min_separation_angle=45, 
                         mask=mask, 
                         normalize_peaks=True)

"""
This time we will not use FA as input to EuDX but we will use directly
the maximum peaks of the ODF. 
"""

eu = EuDX(peaks.peak_values, 
          peaks.peak_indices, 
          odf_vertices = sphere.vertices, a_low=0.2)

csa_streamlines = [streamline for streamline in eu]

"""
Clear the scene
"""

fvtk.clear(r)

fvtk.add(r, fvtk.line(csa_streamlines, line_colors(csa_streamlines)))

fvtk.record(r, n_frames=1, out_path='csa_tracking.png', size=(600, 600))

"""
In the previous example we show that in the Tensor case we discretized the
eigen_vectors and for the CSA case we used peaks_from_model as the CSA is a
model that creates ODFs. However, the Tensor creates ODFs too which are always
ellipsoids. So, we could have used the Tensor with peaks_from_model.
"""

