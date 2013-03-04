"""
===========================================
Reconstruct with Diffusion Spectrum Imaging
===========================================

We show how to apply Diffusion Spectrum Imaging (Wedeen et al. Science 2012) to
diffusion MRI datasets of Cartesian keyhole diffusion gradients.

First import the necessary modules:
"""

import nibabel as nib
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.align.aniso2iso import resample
from dipy.reconst.dsi import DiffusionSpectrumModel

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
data.shape ``(96, 96, 60, 203)``

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

"""
Lets just use one slice only from the data.
"""

dataslice = data[:, :, data.shape[2] / 2]

dsfit = dsmodel.fit(dataslice)

"""
Load an odf reconstruction sphere
"""

sphere = get_sphere('symmetric724')

"""
Calculate the ODFs with this specific sphere
"""

ODF = dsfit.odf(sphere)

print('ODF.shape (%d, %d, %d)' % ODF.shape)

"""
ODF.shape ``(96, 96, 724)``

In a similar fashion it is possible to calculate the PDFs of all voxels
in one call with the following way
"""

PDF = dsfit.pdf()

print('PDF.shape (%d, %d, %d, %d, %d)' % PDF.shape)

"""
PDF.shape ``(96, 96, 17, 17, 17)``

We see that even for a single slice this PDF array is close to 345 MBytes so we
really have to be careful with memory usage when use this function with a full
dataset.

The simple solution is to generate/analyze the ODFs/PDFs by iterating through
each voxel and not store them in memory if that is not necessary.
"""

from dipy.core.ndindex import ndindex

for index in ndindex(dataslice.shape[:2]):
    pdf = dsmodel.fit(dataslice[index]).pdf()

"""
If you really want to save the PDFs of a full dataset on the disc we recommend
using memory maps (numpy.memmap) but still have in mind that if you do that for
example for a dataset of volume size ``(96, 96, 60)`` you will need about 20
which can take less space when reasonable spheres (with < 1000 vertices) are.

.. include:: ../links_names.inc

"""
