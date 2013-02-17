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

from dipy.data import fetch_stanford_hardi

"""
Fetch will download the raw dMRI dataset of a single subject. The size of the dataset is 51 MBytes.
You only need to fetch once.
"""

fetch_stanford_hardi()

"""
Next, we read the saved dataset
"""

from dipy.data import read_stanford_hardi

img, gtab = read_stanford_hardi()

"""
img contains a nibabel Nifti1Image object (with the data) and gtab contains a GradientTable
object (information about the gradients e.g. b-values and b-vectors).
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data.shape ``(128, 128, 49, 65)``

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
data2.shape ``(115, 115, 61, 65)``

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
"""

evecs_img = nib.Nifti1Image(tenfit.evecs, img.get_affine())
nib.save(evecs_img, 'tensor_evecs.nii.gz')

"""
Finally lets try to visualize the orientation distribution functions of a small
rectangular area around the middle of our datasets.
"""

i,j,k,w = np.array(data2.shape) / 2
data_small  = data2[i-5:i+5, j-5:j+5, k-2:k+2]
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(tenmodel.fit(data_small).odf(sphere),
							  sphere, colormap=None))

print('Saving illustration as tensor_odfs.png')
fvtk.record(r, n_frames=1, out_path='tensor_odfs.png', size=(600, 600))

"""
.. figure:: tensor_odfs.png
   :align: center

   **Tensor ODFs**.

.. include:: ../links_names.inc

"""
