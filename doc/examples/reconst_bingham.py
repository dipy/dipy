"""
===========================================================================
Reconstruction of Bingham Functions from ODFs
===========================================================================

This example shows how to recontruct Bingham functions from orientation
distribution functions (ODFs). Reconstructed Bingham functions can be useful to
quantify properties from ODFs such as fiber dispersion [1]_,[2]_.

To begin, let's import the relevant functions and load a data consisting of 10
b0s and 150 non-b0s with a b-value of 2000.
"""

import matplotlib.pyplot as plt

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import (auto_response_ssst,
                                   ConstrainedSphericalDeconvModel)
from dipy.direction.bingham import bingham_from_odf
from dipy.viz import window, actor


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

sphere = get_sphere('repulsion724')
sphere = sphere.subdivide(2)

###############################################################################
# Step 1. ODF estimation
# =================================================
#
# Before fitting Bingham functions, we need to reconstruct orientation
# distribution functions. In this example, ODFs will be reconstructed using
# the Constrained Spherical Deconvolution (CSD) method [3]_.
# In the main tutorial of CSD (see
# :ref:`sphx_glr_examples_built_reconstruction_reconst_csd.py`), several
# strategies to define the fiber response function are discussed. Here, for
# the sake of simplicity, the use the response function estimates from a local
# brain regions:

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

# Let's now compute the ODFs using this response function:

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

###############################################################################
# For efficiency, we will only fit a small part of the data.

data_small = data[20:50, 55:85, 38:39]
csd_fit = csd_model.fit(data_small)

###############################################################################
# Let's visualize the odfs

csd_odf = csd_fit.odf(sphere)

interactive = False

scene = window.Scene()

fodf_spheres = actor.odf_slicer(csd_odf, sphere=sphere, scale=0.9,
                                norm=False, colormap='plasma')
scene.add(fodf_spheres)

print('Saving illustration as csd_odfs.png')
window.record(scene, out_path='csd_odfs.png', size=(600, 600))
if interactive:
    window.show(scene)


# Step 2. Bingham fitting and Metrics
# =================================================
# Now that we have some ODFs, let's fit the Bingham functions to them by using
# the function `bingham_from_odf`:

BinghamMetrics = bingham_from_odf(csd_odf, sphere)

# The above function outputs a `BinghamMetrics` class instance, containing the
# parameters of the fitted Bingham functions. For instance, the fitted Bingham
# functions can be visualized using the following lines of code:

bim_odf = BinghamMetrics.odf(sphere)

scene.rm(fodf_spheres)

fodf_spheres = actor.odf_slicer(bim_odf, sphere=sphere, scale=0.9,
                                norm=False, colormap='plasma')
scene.add(fodf_spheres)

print('Saving illustration as Bingham_odfs.png')
window.record(scene, out_path='Bingham_odfs.png', size=(600, 600))
if interactive:
    window.show(scene)

# As mentioned above, reconstructed Bingham functions can be useful to
# quantify properties from ODFs [1]_, [2]_. Below we plot the Bingham metrics
# expected to be proportional to the fiber density (FD) of specific fiber
# populations.

FD_ODF_l1 = BinghamMetrics.fd[:, :, 0, 0]
FD_ODF_l2 = BinghamMetrics.fd[:, :, 0, 1]
FD_total = BinghamMetrics.tfd[:, :, 0]

fig1, ax = plt.subplots(1, 3, figsize=(16, 4))

im0 = ax[0].imshow(FD_ODF_l1[:, -1:1:-1].T, vmin=0, vmax=2)
ax[0].set_title('FD ODF lobe 1')

im1 = ax[1].imshow(FD_ODF_l2[:, -1:1:-1].T, vmin=0, vmax=2)
ax[1].set_title('FD ODF lobe 2')

im2 = ax[2].imshow(FD_total[:, -1:1:-1].T, vmin=0, vmax=2)
ax[2].set_title('FD total')

fig1.colorbar(im0, ax=ax[0])
fig1.colorbar(im1, ax=ax[1])
fig1.colorbar(im2, ax=ax[2])

# From left to right, the above figure shows: 1) the FD estimated from the
# first ODF peak (showing larger values in white matter); 2) the FD estimated
# from the second ODF peak (showing non-zero values in regions of crossing
# white matter fibers); and 3) the sum of FD estimates across all ODF lobes
# (quantity that should be proportional to the density of all fibers within
# each voxel).
#
# Bingham function can also be used to quantify fiber dispersion from the
# ODFs [2]_. Below we show how to extract three orientation dispersion indexes
# (ODI) from the largest ODF peak that should be related to the white matter
# fiber populations occupying larger volume. Note that, for better
# visualization of ODI estimates, voxels with total FD lower than 0.5 are
# masked.

ODIt = BinghamMetrics.odi_total[:, :, 0, 0]
ODI1 = BinghamMetrics.odi_1[:, :, 0, 0]
ODI2 = BinghamMetrics.odi_2[:, :, 0, 0]

ODIt[FD_total < 0.5] = 0
ODI1[FD_total < 0.5] = 0
ODI2[FD_total < 0.5] = 0

fig2, ax = plt.subplots(1, 3, figsize=(15, 5))

im0 = ax[0].imshow(ODI1[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[0].set_title('ODI1 (lobe 1)')

im1 = ax[1].imshow(ODI2[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[1].set_title('ODI2  (lobe 1)')

im2 = ax[2].imshow(ODIt[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[2].set_title('ODIt  (lobe 1)')

fig2.colorbar(im0, ax=ax[0])
fig2.colorbar(im1, ax=ax[1])
fig2.colorbar(im2, ax=ax[2])

# Bingham functions allows the quantification of dispersion across two main
# axes, offering unique information of fiber orientation variability within
# brain tissue [2]_.  Indeed, the figure above shows from left to right:
# 1) ODI for the Axis with Greater Dispersion: This showcases the ODI values
# where fibers exhibit the most variability in orientation.
# 2) ODI for the Axis with Lesser Dispersion: Contrasts with the first by
# highlighting the axis along which fiber orientations are more uniform.
# 3) Average ODI Across Both Axes: Provides a averaged measure of dispersion by
# averaging ODI values from both principal axes of dispersion.
#
# Above, we focused on the largest ODF's lobe, representing the most pronounced
# fiber population within a voxel. However, this methodology is not limited to
# a singular lobe since it can be applied to the other ODF lobes. Below, we
# show the analogous figures  for the second-largest ODF lobe.

ODIt = BinghamMetrics.odi_total[:, :, 0, 1]
ODI1 = BinghamMetrics.odi_1[:, :, 0, 1]
ODI2 = BinghamMetrics.odi_2[:, :, 0, 1]

ODIt[FD_total < 0.5] = 0
ODI1[FD_total < 0.5] = 0
ODI2[FD_total < 0.5] = 0

fig3, ax = plt.subplots(1, 3, figsize=(15, 5))

im0 = ax[0].imshow(ODI1[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[0].set_title('ODI1 (lobe 2)')

im1 = ax[1].imshow(ODI2[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[1].set_title('ODI2  (lobe 2)')

im2 = ax[2].imshow(ODIt[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[2].set_title('ODIt  (lobe 2)')

fig3.colorbar(im0, ax=ax[0])
fig3.colorbar(im1, ax=ax[1])
fig3.colorbar(im2, ax=ax[2])

# Note that in the later figure, regions of white matter that contain only a
# single fiber population display ODI estimates of zero, corresponding to
# ODF profiles lacking a second ODF lobe.
#
# BinghamMetric can also be used to compute the averaged ODI quantities across
# all ODF lobes (see below). The averaged quantitaties are computed by
# weigthing each ODF lobe with their respective FD value.

ODIt = BinghamMetrics.godi_total[:, :, 0]
ODI1 = BinghamMetrics.godi_1[:, :, 0]
ODI2 = BinghamMetrics.godi_2[:, :, 0]

ODIt[FD_total < 0.5] = 0
ODI1[FD_total < 0.5] = 0
ODI2[FD_total < 0.5] = 0

fig4, ax = plt.subplots(1, 3, figsize=(15, 5))

im0 = ax[0].imshow(ODI1[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[0].set_title('ODI1 (global)')

im1 = ax[1].imshow(ODI2[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[1].set_title('ODI2  (global)')

im2 = ax[2].imshow(ODIt[:, -1:1:-1].T, vmin=0, vmax=0.2)
ax[2].set_title('ODIt  (global)')

fig4.colorbar(im0, ax=ax[0])
fig4.colorbar(im1, ax=ax[1])
fig4.colorbar(im2, ax=ax[2])

###############################################################################
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
