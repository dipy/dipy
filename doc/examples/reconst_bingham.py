"""
===========================================================================
Reconstruction of Bingham Functions from ODFs
===========================================================================

This example shows how to recontruct Bingham functions from orientation
distribution functions (ODFs). Reconstructed Bingham functions can be useful to
quantify properties from ODFs such as fiber dispersion [1]_,[2]_.

To begin, let us import the relevant functions and load a data consisting of 10
b0s and 150 non-b0s with a b-value of 2000s/mm2.
"""

import matplotlib.pyplot as plt

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   ConstrainedSphericalDeconvModel)
from dipy.direction.bingham import (bingham_from_odf, bingham_from_sh)
from dipy.viz import window, actor


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

###############################################################################
# To properly fit Bingham functions, we recommend the use of a larger number of
# directions to sample the ODFs. For this, we load a `sphere` class instance
# containing 724 directions sampling a 3D sphere. We further subdivide the
# faces of this `sphere` representation into 2, to get 11554 directions.

sphere = get_sphere('repulsion724')
sphere = sphere.subdivide(2)

nd = sphere.vertices.shape[0]
print('The number of directions on the sphere is {}'.format(nd))

###############################################################################
# Step 1. ODF estimation
# =================================================
#
# Before fitting Bingham functions, we must reconstruct ODFs. In this example,
# fiber ODFs (fODFs) will be reconstructed using the Constrained Spherical
# Deconvolution (CSD) method [3]_. For simplicity, we will refer to fODFs
# as ODFs.
# In the main tutorial of CSD (see
# :ref:`sphx_glr_examples_built_reconstruction_reconst_csd.py`), several
# strategies to define the fiber response function are discussed. Here, for
# the sake of simplicity, we will use the response function estimates from a
# local brain regions:

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

# Let us now compute the ODFs using this response function:

csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)

###############################################################################
# For efficiency, we will only fit a small part of the data.

data_small = data[20:50, 55:85, 38:39]
csd_fit = csd_model.fit(data_small)

###############################################################################
# Let us visualize the ODFs

csd_odf = csd_fit.odf(sphere)

interactive = False

scene = window.Scene()

fodf_spheres = actor.odf_slicer(csd_odf, sphere=sphere, scale=0.9,
                                norm=False, colormap='plasma')
scene.add(fodf_spheres)

print('Saving the illustration as csd_odfs.png')
window.record(scene, out_path='csd_odfs.png', size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Fiber ODFs (fODFs) reconstructed using Constrained Spherical
# Deconvolution (CSD). For simplicity, we will refer to them just as ODFs.
#
#
# Step 2. Bingham fitting and Metrics
# =================================================
# Now that we have some ODFs, let us fit the Bingham functions to them by using
# the function `bingham_from_odf`:

BinghamMetrics = bingham_from_odf(csd_odf, sphere)

###############################################################################
# The above function outputs a `BinghamMetrics` class instance, containing the
# parameters of the fitted Bingham functions. The metrics of interest contained
# in the `BinghamMetrics` class instance are:
#
# - afd (apparent fiber density: the maximum value for each peak. Also known as
#   Bingham's f_0 parameter.)
# - fd (fiber densitiy: as defined in [1]_, one for each peak.)
# - fs (fiber spread: as defined in [1]_, one for each peak.)
# - gfd (global fiber density: average of fd across all ODF peaks.)
# - gfs (global fiber spread: average of fs across all ODF peaks.)
# - odi_1 (orientation dispersion index along Bingham's first dispersion axis,
#       one for each peak. Defined in [2]_ and [4]_.)
# - odi_2 (orientation dispersion index along Bingham's second dispersion axis,
#       one for each peak.)
# - odi_total (orientation dispersion index averaged across both Bingham's
#       dispersion axes. Defined in [5]_.)
# - godi_1 (global dispersion index along Bingham's first dispersion axis,
#       averaged across all peaks)
# - godi_2 (global dispersion index along Bingham's second dispersion axis,
#       averaged across all peaks)
# - godi_total (global dispersion index averaged across both Bingham's axes,
#       averaged across all peaks)
# - peak_dirs (peak directions in cartesian coordinates given by the Bingham
#       fitting, also known as parameter mu_0. These directions are slightly
#       different than the peak directions given by the function
#       `peaks_from_model`.)
# 
# For illustration purposes, the fitted Bingham derived metrics can be 
# visualized using the following lines of code:

bim_odf = BinghamMetrics.odf(sphere)

scene.rm(fodf_spheres)

fodf_spheres = actor.odf_slicer(bim_odf, sphere=sphere, scale=0.9,
                                norm=False, colormap='plasma')
scene.add(fodf_spheres)

print('Saving the illustration as Bingham_odfs.png')
window.record(scene, out_path='Bingham_odfs.png', size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Bingham functions fitted to CSD fiber ODFs.
#
# Alternatively to fitting Bingham functions to sampled ODFs, DIPY also
# contains the function `bingham_from_sh` to perform Bingham fitting from the
# ODF's spherical harmonic representation. Although this process may require
# longer processing times, this function may be useful to avoid memory issues
# in handling heavily sampled ODFs. Below we show the lines of code to use
# function `bingham_from_sh` (feel free to skip these lines if the function
# `bingham_from_odf` worked fine for you). Note, to use `bingham_from_sh` you
# need to specify the maximum order of spherical harmonics that you defined in
# `csd_model` (in this example this was set to 8):

sh_coeff = csd_fit.shm_coeff
BinghamMetrics = bingham_from_sh(sh_coeff, sphere, 8)

###############################################################################
# Step 3. Bingham Metrics
# =================================================
# As mentioned above, reconstructed Bingham functions can be useful to
# quantify properties from ODFs [1]_, [2]_. Below we plot the Bingham metrics
# expected to be proportional to the fiber density (FD) of specific fiber
# populations.

FD_ODF_l1 = BinghamMetrics.fd[:, :, 0, 0]
FD_ODF_l2 = BinghamMetrics.fd[:, :, 0, 1]
FD_total = BinghamMetrics.gfd[:, :, 0]

fig1, ax = plt.subplots(1, 3, figsize=(16, 4))

im0 = ax[0].imshow(FD_ODF_l1[:, -1:1:-1].T, vmin=0, vmax=2)
ax[0].set_title('FD ODF lobe 1')

im1 = ax[1].imshow(FD_ODF_l2[:, -1:1:-1].T, vmin=0, vmax=2)
ax[1].set_title('FD ODF lobe 2')

im2 = ax[2].imshow(FD_total[:, -1:1:-1].T, vmin=0, vmax=2)
ax[2].set_title('FD ODF global')

fig1.colorbar(im0, ax=ax[0])
fig1.colorbar(im1, ax=ax[1])
fig1.colorbar(im2, ax=ax[2])

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# The figure shows from left to right: 1) the FD estimated for the
# first ODF peak (showing larger values in white matter); 2) the FD estimated
# for the second ODF peak (showing non-zero values in regions of crossing
# white matter fibers); and 3) the sum of FD estimates across all ODF lobes
# (quantity that should be proportional to the density of all fibers within
# each voxel).
#
# Bingham functions can also be used to quantify fiber dispersion from the
# ODFs [2]_. Additionaly to quantifying a combined orientation dispersion
# index (`ODI_total`) for each ODF lobe [5]_, Bingham functions allow  the
# quantification of dispersion across two main axes (`ODI_1` and `ODI_2`),
# offering unique information of fiber orientation variability within the brain
# tissue. Below we show how to extract these indexes from the largest ODF peak.
# Note, for better visualization of ODI estimates, voxels with total FD lower
# than 0.5 are masked.

ODIt = BinghamMetrics.odi_total[:, :, 0, 0]
ODI1 = BinghamMetrics.odi_1[:, :, 0, 0]
ODI2 = BinghamMetrics.odi_2[:, :, 0, 0]

ODIt[FD_total < 0.5] = 0
ODI1[FD_total < 0.5] = 0
ODI2[FD_total < 0.5] = 0

fig2, ax = plt.subplots(1, 3, figsize=(15, 5))

im0 = ax[0].imshow(ODI1[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[0].set_title('ODI_1 (lobe 1)')

im1 = ax[1].imshow(ODI2[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[1].set_title('ODI_2 (lobe 1)')

im2 = ax[2].imshow(ODIt[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[2].set_title('ODI_total (lobe 1)')

fig2.colorbar(im0, ax=ax[0])
fig2.colorbar(im1, ax=ax[1])
fig2.colorbar(im2, ax=ax[2])

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# The figure shows from left to right: 1) ODI of the largest ODF lobe along
# the axis with greater dispersion (direction in which fibers exhibit the most
# variability in orientation); 2) ODI of the largest ODF lobe along the axis
# with lesser dispersion (directions in which fiber orientations are more
# uniform); and 3) total ODI of the largest ODF lobe across both axes.
#
# Above, we focused on the largest ODF's lobe, representing the most pronounced
# fiber population within a voxel. However, this methodology is not limited to
# a singular lobe since it can be applied to the other ODF lobes. Below, we
# show the analogous figures for the second-largest ODF lobe. Note that for
# this figure, regions of white matter that contain only a single fiber
# population display ODI estimates of zero, corresponding to ODF profiles
# lacking a second ODF lobe.

ODIt = BinghamMetrics.odi_total[:, :, 0, 1]
ODI1 = BinghamMetrics.odi_1[:, :, 0, 1]
ODI2 = BinghamMetrics.odi_2[:, :, 0, 1]

ODIt[FD_total < 0.5] = 0
ODI1[FD_total < 0.5] = 0
ODI2[FD_total < 0.5] = 0

fig3, ax = plt.subplots(1, 3, figsize=(15, 5))

im0 = ax[0].imshow(ODI1[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[0].set_title('ODI_1 (lobe 2)')

im1 = ax[1].imshow(ODI2[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[1].set_title('ODI_2 (lobe 2)')

im2 = ax[2].imshow(ODIt[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[2].set_title('ODI_total (lobe 2)')

fig3.colorbar(im0, ax=ax[0])
fig3.colorbar(im1, ax=ax[1])
fig3.colorbar(im2, ax=ax[2])

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# The figure shows from left to right: 1) ODI for the second-largest ODF lobe
# along the axis with greater dispersion (direction in which fibers exhibit the
# most variability in orientation); 2) ODI for the second-largest ODF lobe
# along the axis with lesser dispersion (directions in which fiber
# orientations are more uniform); and 3) total ODI for the second-largest ODF
# lobe across both axes. In this figure, regions of the white matter that
# contain only a single fiber population (one ODF lobe) display ODI estimates
# of zero, corresponding to ODF profiles lacking a second ODF lobe.
#
# BinghamMetrics can also be used to compute the average ODI quantities across
# all ODF lobes aka. global ODI (see below). The average quantitaties are
# computed by weigthing each ODF lobe with their respective FD value.
# These quantities are plotted in the following figure.

ODIt = BinghamMetrics.godi_total[:, :, 0]
ODI1 = BinghamMetrics.godi_1[:, :, 0]
ODI2 = BinghamMetrics.godi_2[:, :, 0]

ODIt[FD_total < 0.5] = 0
ODI1[FD_total < 0.5] = 0
ODI2[FD_total < 0.5] = 0

fig4, ax = plt.subplots(1, 3, figsize=(15, 5))

im0 = ax[0].imshow(ODI1[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[0].set_title('ODI_1 (global)')

im1 = ax[1].imshow(ODI2[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[1].set_title('ODI_2 (global)')

im2 = ax[2].imshow(ODIt[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[2].set_title('ODI_total (global)')

fig4.colorbar(im0, ax=ax[0])
fig4.colorbar(im1, ax=ax[1])
fig4.colorbar(im2, ax=ax[2])

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# The figure shows from left to right: 1) weighted-averaged ODI_1 along all ODF
# lobes; 2) weighted-averaged ODI_2 along all ODF lobe; 3) weighted-averaged
# ODI_total across all ODF lobes.
#
# References
# ----------
#
# .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
#        anisotropy: Extraction of bundle-specific structural metrics from
#        crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
# .. [2] R. Neto Henriques, “Advanced methods for diffusion MRI data analysis
#        and their application to the healthy ageing brain.” Apollo -
#        University of Cambridge Repository, 2018. doi: 10.17863/CAM.29356.
# .. [3] J-D. Tournier, F. Calamante and A. Connelly, “Robust determination of
#        the fibre orientation distribution in diffusion MRI: Non-negativity
#        constrained super-resolved spherical deconvolution”, Neuroimage, vol.
#        35, no. 4, pp. 1459-1472, (2007).
# .. [4] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
#        NODDI: practical in vivo neurite orientation dispersion and
#        density imaging of the human brain. Neuroimage. 2012; 61(4),
#        1000-1016. doi: 10.1016/j.neuroimage.2012.03.072
# .. [5] Tariq M, Schneider T, Alexander DC, Wheeler-Kingshott CAG,
#        Zhang H. Bingham–NODDI: Mapping anisotropic orientation dispersion
#        of neurites using diffusion MRI NeuroImage. 2016; 133:207-223.
