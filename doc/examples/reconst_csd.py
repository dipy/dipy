"""

.. _reconst-csd:

===================================================================
Reconstruction with Constrained Spherical Deconvolution model (CSD)
===================================================================

This example shows how to use Constrained Spherical Deconvolution (CSD)
introduced by :footcite:p:`Tournier2007`.

This method is mainly useful with datasets with gradient directions acquired on
a spherical grid.

The basic idea with this method is that if we could estimate the response
function of a single fiber then we could deconvolve the measured signal and
obtain the underlying fiber distribution.

In this way, the reconstruction of the fiber orientation distribution function
(fODF) in CSD involves two steps:

    1. Estimation of the fiber response function
    2. Use the response function to reconstruct the fODF

Let's first load the data. We will use a dataset with 10 b0s and 150 non-b0s
with b-value 2000.
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames, get_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    auto_response_ssst,
    mask_for_response_ssst,
    recursive_response,
    response_from_mask_ssst,
)
from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity
from dipy.sims.voxel import single_tensor_odf
from dipy.viz import actor, window
from dipy.viz.plotting import image_mosaic

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

###############################################################################
# You can verify the b-values of the dataset by looking at the attribute
# ``gtab.bvals``. Now that a dataset with multiple gradient directions is
# loaded, we can proceed with the two steps of CSD.
#
# Step 1. Estimation of the fiber response function
# =================================================
#
# There are many strategies to estimate the fiber response function. Here two
# different strategies are presented.
#
# **Strategy 1 - response function estimates from a local brain region**
# One simple way to estimate the fiber response function is to look for regions
# of the brain where it is known that there are single coherent fiber
# populations. For example, if we use a ROI at the center of the brain, we will
# find single fibers from the corpus callosum. The ``auto_response_ssst``
# function will calculate FA for a cuboid ROI of radii equal to ``roi_radii``
# in the center of the volume and return the response function estimated in
# that region for the voxels with FA higher than 0.7.

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

###############################################################################
# Note that the ``auto_response_ssst`` function calls two functions that can be
# used separately. First, the function ``mask_for_response_ssst`` creates a
# mask of voxels within the cuboid ROI that meet the FA threshold constraint.
# This mask can be used to calculate the number of voxels that were kept, or
# to also apply an external mask (a WM mask for example). Second, the function
# ``response_from_mask_ssst`` takes the mask and returns the response function
# calculated within the mask. If no changes are made to the mask between the
# two calls, the resulting responses should be identical.

mask = mask_for_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
nvoxels = np.sum(mask)
print(nvoxels)

response, ratio = response_from_mask_ssst(gtab, data, mask)

###############################################################################
# The ``response`` tuple contains two elements. The first is an array with
# the eigenvalues of the response function and the second is the average S0 for
# this response.
#
# It is good practice to always validate the result of auto_response_ssst. For
# this purpose we can print the elements of ``response`` and have a look at
# their values.

print(response)

###############################################################################
# The tensor generated from the response must be prolate (two smaller
# eigenvalues should be equal) and look anisotropic with a ratio of second to
# first eigenvalue of about 0.2. Or in other words, the axial diffusivity of
# this tensor should be around 5 times larger than the radial diffusivity.

print(ratio)

###############################################################################
# We can double-check that we have a good response function by visualizing the
# response function's ODF. Here is how you would do that:

evals = response[0]
evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T


response_sphere = get_sphere(name="symmetric724")
response_odf = single_tensor_odf(response_sphere.vertices, evals=evals, evecs=evecs)

scene = window.Scene()
scene.add(
    actor.odf_slicer(sh_func=response_odf[None, None, None, :], sphere=response_sphere)
)
show_manager = window.ShowManager(title="csd_response", size=(200, 200), scene=scene)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Single-fiber response ODF, normalized.

###############################################################################
# **Strategy 2 - data-driven calibration of response function** Depending
# on the dataset, FA threshold may not be the best way to find the best
# possible response function. For one, it depends on the diffusion tensor
# (FA and first eigenvector), which has lower accuracy at high
# b-values. Alternatively, the response function can be calibrated in a
# data-driven manner :footcite:p:`Tax2014`.
#
# First, the data is deconvolved with a 'fat' response function. All voxels
# that are considered to contain only one peak in this deconvolution (as
# determined by the peak threshold which gives an upper limit of the ratio
# of the second peak to the first peak) are maintained, and from these voxels
# a new response function is determined. This process is repeated until
# convergence is reached. Here we calibrate the response function on a small
# part of the data.

###############################################################################
# A WM mask can shorten computation time for the whole dataset. Here it is
# created based on the DTI fit.

tenmodel = TensorModel(gtab)
brain_mask = data[..., 0] > 200
tenfit = tenmodel.fit(data, mask=brain_mask)

FA = fractional_anisotropy(tenfit.evals)
MD = mean_diffusivity(tenfit.evals)
wm_mask = np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011)))

response = recursive_response(
    gtab,
    data,
    mask=wm_mask,
    sh_order_max=8,
    peak_thr=0.01,
    init_fa=0.08,
    init_trace=0.0021,
    iter=8,
    convergence=0.001,
    parallel=True,
    num_processes=2,
)


###############################################################################
# We can check the shape of the signal of the response function, which should
# be like  a pancake:

response_signal = response.on_sphere(response_sphere)
scene.clear()
scene.add(
    actor.odf_slicer(
        sh_func=response_signal[None, None, None, :],
        sphere=response_sphere,
        scale=0.7,
    )
)
show_manager = window.ShowManager(
    title="csd_recursive_response", size=(200, 200), scene=scene
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Recursively calibrated response signal, normalized.

###############################################################################
# Step 2. fODF reconstruction
# ===========================
#
# After estimating a response function for one of the strategies shown above,
# we are ready to start the deconvolution process. Let's import the CSD model
# and fit the datasets.

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

###############################################################################
# Fit a full axial brain slice for AFD; crop only the ODF and peak displays.

data_slice = data[:, :, 38:39]
csd_fit = csd_model.fit(data_slice, mask=brain_mask[:, :, 38:39])

###############################################################################
# Visualize the CSD fiber ODFs once, directly from the fitted SH coefficients,
# in this 30 x 30 region.

fodf_spheres = actor.odf_slicer(
    sh_coeffs=csd_fit.shm_coeff[20:50, 55:85],
    basis_type="descoteaux07_legacy",
    scale=0.9,
    norm=False,
    visibility=(False, False, True),
)
scene.clear()
scene.add(fodf_spheres)
show_manager = window.ShowManager(title="csd_odfs", size=(600, 600), scene=scene)
window.set_camera_from_axis(show_manager.screens[0], (0, 0, 1))
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# CSD ODFs with orientation colors and preserved relative amplitudes.
#
#
# In DIPY we also provide tools for finding the peak directions (maxima) of the
# ODFs. For this purpose we recommend using ``peaks_from_model``.

csd_peaks = peaks_from_model(
    model=csd_model,
    data=data_slice[20:50, 55:85],
    sphere=default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    parallel=True,
    num_processes=2,
)

fodf_peaks = actor.peaks_slicer(
    csd_peaks.peak_dirs,
    peak_values=csd_peaks.peak_values,
    visibility=(False, False, True),
)
scene.clear()
scene.add(fodf_peaks)
show_manager = window.ShowManager(title="csd_peaks", size=(600, 600), scene=scene)
window.set_camera_from_axis(show_manager.screens[0], (0, 0, 1))
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# CSD Peaks.
#
# Apparent fiber density
# ======================
#
# Apparent fiber density (AFD) :footcite:p:`Raffelt2012` is the integral over
# each segmented FOD lobe :footcite:p:`Smith2013`. ``afd`` returns up to five
# lobes, ordered by peak amplitude, not AFD. Missing lobes are zero; exclude
# them from the minimum.
#
# The maps summarize the retained lobes, not peak heights or tissue fractions.
# AFD and its absolute thresholds depend on the response/FOD scaling.

csd_afd = csd_fit.afd(npeaks=5)
afd_max = csd_afd.max(axis=-1)
afd_min = np.min(csd_afd, axis=-1, where=csd_afd > 0, initial=np.inf)
afd_min[afd_max == 0] = 0
afd_sum = csd_afd.sum(axis=-1)

_fig, _axes = image_mosaic(
    [np.rot90(metric[:, :, 0]) for metric in (afd_min, afd_max, afd_sum)],
    ax_labels=["Minimum lobe AFD", "Maximum lobe AFD", "Summed AFD"],
    ax_kwargs=[{"cmap": "viridis", "vmin": 0, "vmax": afd_sum.max()}] * 3,
    figsize=(13, 4),
    filename="csd_afd.png",
)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Minimum nonzero, maximum, and summed lobe AFD across the full axial brain slice.
# Voxels with no retained lobes are zero.

###############################################################################
#
# References
# ----------
#
# .. footbibliography::
#
