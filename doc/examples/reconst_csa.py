"""
==============================================
Reconstruct with Constant Solid Angle (Q-Ball)
==============================================

We show how to apply a Constant Solid Angle ODF (Q-Ball) model from
:footcite:t:`Aganj2010` to your datasets.

First import the necessary modules:
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.shm import CsaOdfModel, sf_to_sh
from dipy.segment.mask import bounding_box, crop, median_otsu
from dipy.viz.skyline import skyline

###############################################################################
# Download and read the data for this tutorial and load the raw diffusion data
# and the affine.

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

###############################################################################
# img contains a nibabel Nifti1Image object (data) and gtab contains a
# GradientTable object (gradient information e.g. b-values). For example to
# read the b-values it is possible to write print(gtab.bvals).

print(f"data.shape {data.shape}")

###############################################################################
# Remove most of the background using DIPY's mask module.

maskdata, mask = median_otsu(
    data, vol_idx=range(10, 50), median_radius=3, numpass=1, dilate=2
)

"""
We crop the data and mask to the smallest possible region that contains all
the non-zero voxels. The ``crop`` function requires the volume to crop and
the minimum and maximum indices for each dimension.
"""

mins, maxs = bounding_box(mask)
maskdata = crop(maskdata, mins, maxs)
mask = crop(mask, mins, maxs)

###############################################################################
# We instantiate our CSA model with spherical harmonic order ($l$) of 4

csamodel = CsaOdfModel(gtab, 4)

###############################################################################
# `Peaks_from_model` is used to calculate properties of the ODFs (Orientation
# Distribution Function) and return for
# example the peaks and their indices, or GFA which is similar to FA but for
# ODF based models. This function mainly needs a reconstruction model, the
# data and a sphere as input. The sphere is an object that represents the
# spherical discrete grid where the ODF values will be evaluated.

csapeaks = peaks_from_model(
    model=csamodel,
    data=maskdata,
    sphere=default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=mask,
    return_odf=False,
    normalize_peaks=True,
)

GFA = csapeaks.gfa

print(f"GFA.shape {GFA.shape}")

###############################################################################
# Apart from GFA, csapeaks also has the attributes peak_values, peak_indices
# and ODF. peak_values shows the maxima values of the ODF and peak_indices
# gives us their position on the discrete sphere that was used to do the
# reconstruction of the ODF. In order to obtain the full ODF, return_odf
# should be True. Before enabling this option, make sure that you have enough
# memory.
#
# Let's visualize the ODFs of a small rectangular area in an axial slice of the
# splenium of the corpus callosum (CC).

data_small = maskdata[13:43, 44:74, 28:29]

csaodfs = csamodel.fit(data_small).odf(default_sphere)

###############################################################################
# It is common with CSA ODFs to produce negative values, we can remove those
# using ``np.clip``

csaodfs = np.clip(csaodfs, 0, np.max(csaodfs, -1)[..., None])
sh_coeffs = sf_to_sh(
    csaodfs,
    default_sphere,
    sh_order_max=4,
    basis_type="descoteaux07",
    full_basis=False,
    legacy=False,
)

################################################################################
# Let's scale the ODFs to make them more visible. We will scale the ODFs to a
# maximum of 0.4, but you can change this value to your liking.

odf_max = np.max(np.abs(csaodfs), axis=-1, keepdims=True)
odf_scale = np.divide(0.4, odf_max, out=np.zeros_like(odf_max), where=odf_max > 0)
sh_coeffs *= odf_scale

_skyline_obj = skyline(
    sh_coeffs=[(sh_coeffs, None, "CSA ODFs", "descoteaux07")],
    out_stealth_png="csa_odfs.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Constant Solid Angle ODFs.
#
#
# References
# ----------
#
# .. footbibliography::
#
