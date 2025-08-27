"""
===============================================
Reconstruct with Generalized Q-Sampling Imaging
===============================================

We show how to apply Generalized Q-Sampling Imaging :footcite:p:`Yeh2010`
to diffusion MRI datasets. You can think of GQI as an analytical version of
DSI orientation distribution function (ODF) (Garyfallidis, PhD thesis, 2012).

First import the necessary modules:
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.gqi import GeneralizedQSamplingModel

###############################################################################
# Download and get the data filenames for this tutorial.

fraw, fbval, fbvec = get_fnames(name="taiwan_ntu_dsi")

###############################################################################
# img contains a nibabel Nifti1Image object (data) and gtab contains a
# ``GradientTable`` object (gradient information e.g. b-values). For example
# to read the b-values it is possible to write::
#
#    print(gtab.bvals)
#
# Load the raw diffusion data and the affine.

data, affine, voxel_size = load_nifti(fraw, return_voxsize=True)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvecs[1:] = bvecs[1:] / np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None]
gtab = gradient_table(bvals, bvecs=bvecs)
print(f"data.shape {data.shape}")

###############################################################################
# This dataset has anisotropic voxel sizes, therefore reslicing is necessary.
#
# Instantiate the model and apply it to the data.

gqmodel = GeneralizedQSamplingModel(gtab, sampling_length=3)

###############################################################################
# The parameter ``sampling_length`` is used here to control the diffusion
# sampling length (lambda) in the GQI reconstruction. This parameter affects
# the shape and smoothness of the resulting ODFs. The optimal value typically
# ranges from 1-1.3 for GQI2 method, but higher values can be used depending on
# the dataset characteristics.
#
# Lets just use one slice only from the data.

dataslice = data[:, :, data.shape[2] // 2]

mask = dataslice[..., 0] > 50

gqfit = gqmodel.fit(dataslice, mask=mask)

###############################################################################
# Load an ODF reconstruction sphere

sphere = get_sphere(name="repulsion724")

###############################################################################
# Calculate the ODFs with this specific sphere

ODF = gqfit.odf(sphere)

print(f"ODF.shape {ODF.shape}")

###############################################################################
# Using ``peaks_from_model`` we can find the main peaks of the ODFs and other
# properties.

gqpeaks = peaks_from_model(
    model=gqmodel,
    data=dataslice,
    sphere=sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=mask,
    return_odf=False,
    normalize_peaks=True,
)

gqpeak_values = gqpeaks.peak_values

###############################################################################
# ``gqpeak_indices`` show which sphere points have the maximum values.

gqpeak_indices = gqpeaks.peak_indices

###############################################################################
# It is also possible to calculate GFA.

GFA = gqpeaks.gfa

print(f"GFA.shape {GFA.shape}")

###############################################################################
# With parameter ``return_odf=True`` we can obtain the ODF using
# ``gqpeaks.ODF``

gqpeaks = peaks_from_model(
    model=gqmodel,
    data=dataslice,
    sphere=sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=mask,
    return_odf=True,
    normalize_peaks=True,
)

###############################################################################
# This ODF will be of course identical to the ODF calculated above as long as
# the same data and mask are used.

print(np.sum(gqpeaks.odf != ODF) == 0)

###############################################################################
# The advantage of using ``peaks_from_model`` is that it calculates the ODF
# only once and saves it or deletes if it is not necessary to keep.
#
#
# References
# ----------
#
# .. footbibliography::
#
