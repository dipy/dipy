"""

=========================
Reconstruct with a Tensor
=========================

This example shows how to reconstruct your diffusion datasets using a single Tensor model.

First import the necessary modules:

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
``dipy.data`` is used for small datasets that we use in tests and examples.
"""

from dipy.data import fetch_beijing_dti

"""
Fetch will download the raw dMRI dataset of a single subject. The size of the dataset is 51 MBytes.
You only need to fetch once.
"""

fetch_beijing_dti()

"""
Next, we read the saved dataset
"""

from dipy.data import read_beijing_dti

img, gtab = read_beijing_dti()

"""
img contains a nibabel Nifti1Image object (with the data) and gtab contains a GradientTable
object (information about the gradients e.g. b-values and b-vectors).
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
This dataset has anisotropic voxel sizes, therefore reslicing is necessary
"""

affine = img.get_affine()

"""
Load and show the zooms which hold the voxel size.
"""

zooms = img.get_header().get_zooms()[:3]

"""
The voxel size here is ``(1.79, 1.79, 2.5)``.

We now set the required new voxel size.
"""

new_zooms = (2., 2., 2.)

"""
Start reslicing. Trilinear interpolation is used by default.
"""

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine, zooms, new_zooms)

print('data2.shape (%d, %d, %d, %d)' % data2.shape)

"""
Lets also create a simple mask. This is a quick way to avoid calculating
Tensors on the background of the image. This is because the signal is very low in
these region. A better way would be to extract the brain region using a brain
extraction method. But will skip that for now.
"""

mask = data2[..., 0] > 50

"""
Now that we have prepared the datasets we can go forward with the voxel
reconstruction. First, we instantiate the Tensor model in the following way.
"""

tenmodel = TensorModel(gtab)

"""
Fitting the data is very simple. We just need to call the fit method of the
TensorModel in the following way:
"""

tenfit = tenmodel.fit(data2, mask)

"""
The fit method creates a TensorFit object which contains the fitting parameters
and other attributes of the model. For example we can generate fractional
anisotropy from the eigen values of the single tensor.
"""

from dipy.reconst.dti import fractional_anisotropy

FA = fractional_anisotropy(tenfit.evals)

"""
In the background of the image the fitting will not be accurate there is no
signal and possibly we will find FA values with nans (not a number). We can
easily remove these in the following way.
"""

FA[np.isnan(FA)] = 0

"""
Saving the FA images is very easy using nibabel. We need the FA volume and the
affine matrix which transform the image's coordinates to the world coordinates.
"""

fa_img = nib.Nifti1Image(FA, img.get_affine())
nib.save(fa_img, 'tensor_fa.nii.gz')

"""
You can now see the result with any nifti viewer or check it slice by slice
using matplotlib_'s imshow. In the same way you can save the eigen values the
eigen vectors or any other properties of the Tensor.

.. include:: ../links_names.inc

"""
