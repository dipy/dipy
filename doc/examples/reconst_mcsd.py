"""

.. _reconst-mcsd:

================================================
Reconstruction with Multi-Shell Multi-Tissue CSD
================================================

This example shows how to use Multi-Shell Multi-Tissue Constrained Spherical
Deconvolution (MSMT-CSD) introduced by Tournier et al. [Jeurissen2014]_. This
tutorial goes through the steps involved in implementing the method.

This method provides improved White Matter(WM), Grey Matter (GM), and
Cerebrospinal fluid (CSF) volume fraction maps, which is otherwise
overestimated in the standard CSD (SSST-CSD). This is done by using b-value
dependencies of the different tissue types to estimate ODFs. This method thus
extends the SSST-CSD introduced in [Tournier2007]_.

The reconstruction of the fiber orientation distribution function
(fODF) in MSMT-CSD involves the following steps:
    1. Generate a mask using Median Otsu (optional step)
    2. Denoise the data using MP-PCA (optional step)
    3. Generate  Anisotropic Powermap (if T1 unavailable)
    4. Fit DTI model to the data
    5. Tissue Classification (needs to be at least two classes of tissues)
    6. Estimation of the fiber response function
    7. Use the response function to reconstruct the fODF

First, we import all the modules we need from dipy as follows:
"""

import numpy as np
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import matplotlib.pyplot as plt

from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.mcsd import (auto_response_msmt,
                               mask_for_response_msmt,
                               response_from_mask_msmt)
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.viz import window, actor

from dipy.data import get_sphere, get_fnames
sphere = get_sphere('symmetric724')

###############################################################################
# For this example, we use fetch to download a multi-shell dataset which was
# kindly provided by Hansen and Jespersen (more details about the data are
# provided in their paper [Hansen2016]_). The total size of the downloaded
# data is 192 MBytes, however you only need to fetch it once.

fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

###############################################################################
# For the sake of simplicity, we only select two non-zero b-values for this
# example.

bvals = gtab.bvals
bvecs = gtab.bvecs

sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
data = data[..., sel_b]

###############################################################################
# The gradient table is also selected to have the selected b-values (0, 1000
# and 2000)

gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

###############################################################################
# We make use of the ``median_otsu`` method to generate the mask for the data
# as follows:

b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])

print(data.shape)

###############################################################################
# As one can see from its shape, the selected data contains a total of 67
# volumes of images corresponding to all the diffusion gradient directions
# of the selected b-values and call the ``mppca`` as follows:

denoised_arr = mppca(data, mask=mask, patch_radius=2)

###############################################################################
# Now we will use the denoised array (``denoised_arr``) obtained from ``mppca``
# in the rest of the steps in the tutorial.
#
# As for the next step, we generate the anisotropic powermap introduced by
# [DellAcqua2014]_. To do so, we make use of the Q-ball Model as follows:

qball_model = shm.QballModel(gtab, 8)

###############################################################################
# We generate the peaks from the ``qball_model`` as follows:

peaks = dp.peaks_from_model(model=qball_model, data=denoised_arr,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            sphere=sphere, mask=mask)

ap = shm.anisotropic_power(peaks.shm_coeff)

plt.matshow(np.rot90(ap[:, :, 10]), cmap=plt.cm.bone)
plt.savefig("anisotropic_power_map.png")
plt.close()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Anisotropic Power Map (Axial Slice)

print(ap.shape)

###############################################################################
# The above figure is a visualization of the axial slice of the Anisotropic
# Power Map. It can be treated as a pseudo-T1 for classification purposes
# using the Hidden Markov Random Fields (HMRF) classifier, if the T1 image
# is not available.
#
# As we can see from the shape of the Anisotropic Power Map, it is 3D and can
# be used for tissue classification using HMRF. The
# HMRF needs the specification of the number of classes. For the case of
# MSMT-CSD the ``nclass`` parameter needs to be ``>=2``. In our case, we set
# it to 3: namely corticospinal fluid (csf), white matter (wm) and gray
# matter (gm).

nclass = 3

###############################################################################
# Then, the smoothness factor of the segmentation. Good performance is achieved
# with values between 0 and 0.5.

beta = 0.1

###############################################################################
# We then call the ``TissueClassifierHMRF`` with the parameters specified as
# above:

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)

###############################################################################
# Then, we get the tissues segmentation from the final_segmentation.

csf = np.where(final_segmentation == 1, 1, 0)
gm = np.where(final_segmentation == 2, 1, 0)
wm = np.where(final_segmentation == 3, 1, 0)

###############################################################################
# Now, we want the response function for each of the three tissues and for each
# bvalues. This can be achieved in two different ways. If the case that tissue
# segmentation is available or that one wants to see the tissue masks used to
# compute the response functions, a combination of the functions
# ``mask_for_response_msmt`` and ``response_from_mask`` is needed.
#
# The ``mask_for_response_msmt`` function will return a mask of voxels within a
# cuboid ROI and that meet some threshold constraints, for each tissue and
# bvalue. More precisely, the WM mask must have a FA value above a given
# threshold. The GM mask and CSF mask must have a FA below given thresholds
# and a MD below other thresholds.
#
# Note that for ``mask_for_response_msmt``, the gtab and data should be for
# bvalues under 1200, for optimal tensor fit.

mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data, roi_radii=10,
                                                    wm_fa_thr=0.7,
                                                    gm_fa_thr=0.3,
                                                    csf_fa_thr=0.15,
                                                    gm_md_thr=0.001,
                                                    csf_md_thr=0.0032)

###############################################################################
# If one wants to use the previously computed tissue segmentation in addition
# to the threshold method, it is possible by simply multiplying both masks
# together.

mask_wm *= wm
mask_gm *= gm
mask_csf *= csf

###############################################################################
# The masks can also be used to calculate the number of voxels for each tissue.

nvoxels_wm = np.sum(mask_wm)
nvoxels_gm = np.sum(mask_gm)
nvoxels_csf = np.sum(mask_csf)

print(nvoxels_wm)

###############################################################################
# Then, the ``response_from_mask`` function will return the msmt response
# functions using precalculated tissue masks.

response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                 mask_wm,
                                                                 mask_gm,
                                                                 mask_csf)

###############################################################################
# Note that we can also get directly the response functions by calling the
# ``auto_response_msmt`` function, which internally calls
# ``mask_for_response_msmt`` followed by ``response_from_mask``. By doing so,
# we don't have access to the masks and we might have problems with high
# bvalues tensor fit.

auto_response_wm, auto_response_gm, auto_response_csf = \
    auto_response_msmt(gtab, data, roi_radii=10)

###############################################################################
# As we can see below, adding the tissue segmentation can change the results
# of the response functions.

print("Responses")
print(response_wm)
print(response_gm)
print(response_csf)
print("Auto responses")
print(auto_response_wm)
print(auto_response_gm)
print(auto_response_csf)

###############################################################################
# At this point, there are two options on how to use those response functions.
# We want to create a MultiShellDeconvModel, which takes a response function as
# input. This response function can either be directly in the current format,
# or it can be a MultiShellResponse format, as produced by the
# ``multi_shell_fiber_response`` method. This function assumes a 3 compartments
# model (wm, gm, csf) and takes one response function per tissue per bvalue. It
# is important to note that the bvalues must be unique for this function.

ubvals = unique_bvals_tolerance(gtab.bvals)
response_mcsd = multi_shell_fiber_response(sh_order_max=8,
                                           bvals=ubvals,
                                           wm_rf=response_wm,
                                           gm_rf=response_gm,
                                           csf_rf=response_csf)

###############################################################################
# As mentioned, we can also build the model directly and it will call
# ``multi_shell_fiber_response`` internally. Important note here, the function
# ``unique_bvals_tolerance`` is used to keep only unique bvalues from the gtab
# given to the model, as input for ``multi_shell_fiber_response``. This may
# introduce differences between the calculated response of each method,
# depending on the bvalues given to ``multi_shell_fiber_response`` externally.

response = np.array([response_wm, response_gm, response_csf])
mcsd_model_simple_response = MultiShellDeconvModel(gtab,
                                                   response,
                                                   sh_order_max=8)

###############################################################################
# Note that this technique only works for a 3 compartments model (wm, gm, csf).
# If one wants more compartments, a custom MultiShellResponse object must be
# used. It can be inspired by the ``multi_shell_fiber_response`` method.
#
# Now we build the MSMT-CSD model with the ``response_mcsd`` as input. We then
# call the ``fit`` function to fit one slice of the 3D data and visualize it.

mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
mcsd_fit = mcsd_model.fit(denoised_arr[:, :, 10:11])

###############################################################################
# The volume fractions of tissues for each voxel are also accessible, as well
# as the sh coefficients for all tissues. One can also get each sh tissue
# separately using ``all_shm_coeff`` for each compartment (isotropic) and
# ``shm_coeff`` for white matter.

vf = mcsd_fit.volume_fractions
sh_coeff = mcsd_fit.all_shm_coeff
csf_sh_coeff = sh_coeff[..., 0]
gm_sh_coeff = sh_coeff[..., 1]
wm_sh_coeff = mcsd_fit.shm_coeff

###############################################################################
# The model allows one to predict a signal from sh coefficients. There are two
# ways of doing this.

mcsd_pred = mcsd_fit.predict()
mcsd_pred = mcsd_model.predict(mcsd_fit.all_shm_coeff)

###############################################################################
# From the fit obtained in the previous step, we generate the ODFs which can be
# visualized as follows:

mcsd_odf = mcsd_fit.odf(sphere)

print("ODF")
print(mcsd_odf.shape)
print(mcsd_odf[40, 40, 0])

fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=1,
                                norm=False, colormap='plasma')

interactive = False
scene = window.Scene()
scene.add(fodf_spheres)
scene.reset_camera_tight()

print('Saving illustration as msdodf.png')
window.record(scene, out_path='msdodf.png', size=(600, 600))

if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# MSMT-CSD Peaks and ODFs.
#
#
# References
# ----------
#
# .. [Jeurissen2014] B. Jeurissen, et al., "Multi-tissue constrained spherical
#                     deconvolution for improved analysis of multi-shell
#                     diffusion MRI data." NeuroImage 103 (2014): 411-426.
#
# .. [Tournier2007] J-D. Tournier, F. Calamante and A. Connelly, "Robust
#                     determination of the fibre orientation distribution in
#                     diffusion MRI: Non-negativity constrained super-resolved
#                     spherical deconvolution", Neuroimage, vol. 35, no. 4,
#                     pp. 1459-1472, (2007).
#
# .. [Hansen2016] B. Hansen and SN. Jespersen, " Data for evaluation of fast
#                     kurtosis strategies, b-value optimization and exploration
#                     of diffusion MRI contrast", Scientific Data 3: 160072
#                     doi:10.1038/sdata.2016.72, (2016)
#
# .. [DellAcqua2014] F. Dell'Acqua, et. al., "Anisotropic Power Maps: A
#                     diffusion contrast to reveal low anisotropy tissues from
#                     HARDI data", Proceedings of International Society for
#                     Magnetic Resonance in Medicine. Milan, Italy, (2014).
