"""

=================================================
Reconstruction using Constant Solid Angle (QBall)
=================================================

We show how to apply a Constant Solid Angle ODF (Q-Ball) model from Aganj et.
al to your datasets.

First import the necessary modules:
"""

import nibabel as nib
from dipy.data import fetch_beijing_dti, read_beijing_dti, get_sphere
from dipy.align.aniso2iso import resample
from dipy.reconst.shm import CsaOdfModel, normalize_data
from dipy.reconst.odf import peaks_from_model

"""
Download and read the data for this tutorial.
"""

fetch_beijing_dti()
img, gtab = read_beijing_dti()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values).
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data.shape (128, 128, 49, 65)

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
Which is ``(2.0, 2.0, 2.0)``

Start reslicing. Trilinear interpolation is used by default.
"""

data2, affine2 = resample(data, affine, zooms, new_zooms)

print('data2.shape (%d, %d, %d, %d)' % data2.shape)

"""
data2.shape (115, 115, 61, 65)

Mask out most of the background.
"""
mask = data2[..., 0] > 50

"""
We instantiate our CSA model with sperical harmonic order of 4
"""

csamodel = CsaOdfModel(gtab, 4)

"""
`Peaks_from_model` is used to calculate properties of the ODFs (Orientation
Distribution Function) and return for
example the peaks and their indices, or GFA which is similar to FA but for ODF
based models. This function mainly needs a reconstruction model, the data and a
sphere as input. The sphere is an object that represents the spherical discrete
grid where the ODF values will be evaluated.
"""

sphere = get_sphere('symmetric724')

peaks = peaks_from_model(model=csamodel,
                         data=data2,
                         sphere=sphere,
                         relative_peak_threshold=.8,
                         min_separation_angle=45,
                         mask=mask,
                         normalize_peaks=True)

GFA = peaks.gfa

print('GFA.shape (%d, %d, %d)' % GFA.shape)

"""
GFA.shape (115, 115, 61)

.. include:: ../links_names.inc

"""
