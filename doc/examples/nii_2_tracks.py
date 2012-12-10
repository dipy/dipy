""" 

===================================
Tensor based streamlines
===================================

Overview
========

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

from dipy.reconst.dti import TensorModel

"""
``dipy.tracking`` is for tractography algorithms which create sets of tracks by integrating 
  directionality models across voxels.
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

Start resampling (reslicing). Trilinear interpolation is used by default.
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
EuDX is the fiber tracking algorithm that we use in this example
"""

from dipy.tracking.eudx import EuDX

eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices)

streamlines = [streamline for streamline in eu]

from dipy.viz import fvtk

r=fvtk.ren()

from dipy.viz.colormap import line_colors

fvtk.add(r, fvtk.line(streamlines, line_colors(streamlines)))

print('Saving illustration as tensor_tracks.png')
fvtk.record(r, n_frames=1, out_path='tensor_tracks.png', size=(600, 600))
