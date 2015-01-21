"""
===============================================
Reconstruct with Generalized Q-Sampling Imaging
===============================================

We show how to apply Generalized Q-Sampling Imaging [Yeh2010]_
to diffusion MRI datasets. You can think of GQI as an analytical version of
DSI orientation distribution function (ODF) (Garyfallidis, PhD thesis, 2012).

First import the necessary modules:
"""

import numpy as np
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.direction import peaks_from_model

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

gqmodel = GeneralizedQSamplingModel(gtab, sampling_length=3)

"""
The parameter `sampling_length` is used here to

Lets just use one slice only from the data.
"""

dataslice = data[:, :, data.shape[2] / 2]

mask = dataslice[..., 0] > 50

gqfit = gqmodel.fit(dataslice, mask=mask)

"""
Load an odf reconstruction sphere
"""

sphere = get_sphere('symmetric724')

"""
Calculate the ODFs with this specific sphere
"""

ODF = gqfit.odf(sphere)

print('ODF.shape (%d, %d, %d)' % ODF.shape)

"""
ODF.shape ``(96, 96, 724)``

Using peaks_from_model we can find the main peaks of the ODFs and other
properties.
"""

gqpeaks = peaks_from_model(model=gqmodel,
                           data=dataslice,
                           sphere=sphere,
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           return_odf=False,
                           normalize_peaks=True)

gqpeak_values = gqpeaks.peak_values

"""
gqpeak_indices show which sphere points have the maximum values.
"""

gqpeak_indices = gqpeaks.peak_indices

"""
It is also possible to calculate GFA.
"""

GFA = gqpeaks.gfa

print('GFA.shape (%d, %d)' % GFA.shape)

"""
With parameter `return_odf=True` we can obtain the ODF using gqpeaks.ODF
"""

gqpeaks = peaks_from_model(model=gqmodel,
                           data=dataslice,
                           sphere=sphere,
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           return_odf=True,
                           normalize_peaks=True)

"""
This ODF will be of course identical to the ODF calculated above as long as the same
data and mask are used.
"""

np.sum(gqpeaks.odf != ODF) == 0

"""
True

The advantage of using peaks_from_models is that it calculates the ODF only once and
saves it or deletes if it is not necessary to keep.

.. [Yeh2010] Yeh, F-C et al., Generalized Q-sampling imaging, IEEE
             Transactions on Medical Imaging, vol 29, no 9, 2010.

.. include:: ../links_names.inc

"""
