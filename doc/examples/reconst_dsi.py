"""
===============================================
Reconstruction using Diffusion Spectrum Imaging
===============================================

We show how to apply Diffusion Spectrum Imaging (Wedeen et al. Science 2012) to
diffusion MRI datasets of Cartesian keyhole gradients.

First import the necessary modules:
"""

import nibabel as nib
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.align.aniso2iso import resample
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.odf import peaks_from_model

"""
Download and read the data for this tutorial.
"""

fetch_taiwan_ntu_dsi()
img, gtab = read_taiwan_ntu_dsi()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine.
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
This dataset has anisotropic voxel sizes, therefore reslicing is necessary.
"""

affine = img.get_affine()

"""
Read the voxel size from the image header.
"""

voxel_size = img.get_header().get_zooms()[:3]

"""
Instantiate the Model and apply it to the data.
"""

dsmodel = DiffusionSpectrumModel(gtab)
dsfit = dsmodel.fit(data)

"""
Load an odf reconstruction sphere
"""
sphere = get_sphere('symmetric724')
ODF = dsfit.odf(sphere)

"""
.. include:: ../links_names.inc

"""
