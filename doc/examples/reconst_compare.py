"""

.. _reconst_compare:

===================================================
Compare Reconstructions with Varying Scan Qualities
===================================================

Let's explore the effect of the number of directions and b-values
on the quality of the reconstruction.
"""

import numpy as np
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import matplotlib.pyplot as plt

from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.nn.evac import EVACPlus
from dipy.segment.mask import median_otsu
from dipy.align.imaffine import AffineMap

from dipy.denoise.localpca import mppca
from dipy.align import affine_registration, motion_correction
from dipy.reconst.mcsd import (auto_response_msmt,
                               mask_for_response_msmt,
                               response_from_mask_msmt)
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.viz import window, actor

from dipy.data import get_sphere, get_fnames
sphere = get_sphere('symmetric724')

###############################################################################
# For this example, we use fetch to download a multi-shell dataset which was
# kindly provided by Hansen and Jespersen (more details about the data are
# provided in their paper [Hansen2016]_). The total size of the downloaded
# data is 192 MBytes, however you only need to fetch it once.

fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

###############################################################################
# First let's classify gray matter vs white matter vs csf using the T1.

t1_data, t1_affine = load_nifti(t1_fname)

# compute brainmask, works better than median otsu in this case
evac = EVACPlus()
mask = evac.predict(t1_data, t1_affine)
t1_mask = mask * t1_data

sagittal_slice = 121
plt.imshow(np.rot90(t1_mask[:, :, sagittal_slice]), cmap=plt.cm.bone)
plt.show()

# classify tissue
hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(
    t1_mask, nclasses=3, beta=0.1)

csf = np.where(final_segmentation == 1, 1, 0)
gm = np.where(final_segmentation == 2, 1, 0)
wm = np.where(final_segmentation == 3, 1, 0)


plt.imshow(np.rot90(np.where(gm[:, :, sagittal_slice], 0.5, np.nan)),
           cmap=plt.cm.bone_r, vmin=0, vmax=1)
plt.imshow(np.rot90(np.where(wm[:, :, sagittal_slice], 0.9, np.nan)),
           cmap=plt.cm.bone, vmin=0, vmax=1)
plt.imshow(np.rot90(np.where(csf[:, :, sagittal_slice], 1, np.nan)),
           cmap=plt.cm.cool)
plt.show()

###############################################################################
# Next, load dMRI data.

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

gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

###############################################################################
# Compute a mask on the diffusion data.

maskdata, mask = median_otsu(data, vol_idx=[0], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)

plt.imshow(np.rot90(maskdata[:, :, 9, 0]), cmap=plt.cm.bone)
plt.show()

denoised = mppca(data, mask=mask, patch_radius=4)

mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data, roi_radii=10,
                                                    wm_fa_thr=0.7,
                                                    gm_fa_thr=0.3,
                                                    csf_fa_thr=0.15,
                                                    gm_md_thr=0.001,
                                                    csf_md_thr=0.0032)

###############################################################################
# Next, register T1 to dMRI

t1_reg, reg_affine = affine_registration(
    moving=t1_data, moving_affine=t1_affine,
    static=maskdata[..., 0], static_affine=affine
)

mapping = AffineMap(
    affine=reg_affine,
    domain_grid_shape=data.shape[:-1],
    domain_grid2world=affine,
    codomain_grid_shape=t1_data.shape,
    codomain_grid2world=t1_affine
)

###############################################################################
# Now, transform the masks to diffusion-weighted imaging space.

wm_dwi = mapping.transform(wm, interpolation='nearest')
gm_dwi = mapping.transform(gm, interpolation='nearest')
csf_dwi = mapping.transform(csf, interpolation='nearest')

axial_slice = 9
plt.imshow(np.rot90(data[:, :, axial_slice, 0]), cmap=plt.cm.bone)
plt.imshow(np.rot90(np.where(gm_dwi[:, :, axial_slice], 0.5, np.nan)),
           cmap=plt.cm.bone_r, vmin=0, vmax=1, alpha=0.5)
plt.imshow(np.rot90(np.where(wm_dwi[:, :, axial_slice], 0.9, np.nan)),
           cmap=plt.cm.bone, vmin=0, vmax=1, alpha=0.5)
plt.imshow(np.rot90(np.where(csf_dwi[:, :, axial_slice], 1, np.nan)),
           cmap=plt.cm.cool, alpha=0.5)
plt.show()

###############################################################################
# Now, generate a response function from the data.

response_wm, response_gm, response_csf = response_from_mask_msmt(
    gtab, data, wm_dwi, gm_dwi, csf_dwi)

ubvals = unique_bvals_tolerance(gtab.bvals)
response_mcsd = multi_shell_fiber_response(sh_order_max=8,
                                           bvals=ubvals,
                                           wm_rf=response_wm,
                                           gm_rf=response_gm,
                                           csf_rf=response_csf)

mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
mcsd_fit = mcsd_model.fit(data)





response, ratio = auto_response_ssst(
    gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(
    data, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=wm_mask)

csa_model = CsaOdfModel(gtab, sh_order_max=6)
gfa = csa_model.fit(data, mask=wm_mask).gfa
shm_coeff = csa_model.shm_coeff


tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data)
FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

fig, axs = plt.subplots(1, 3)
for i, ax in enumerate(axs):
    ax.axis('off')
    ax.imshow(FA[(slice(None),) * i + (FA.shape[i] // 2,)])
fig.show()

# tensor_odfs = fit.odf(default_sphere)
pam = peaks_from_model(
    tenmodel, data, default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=wm_mask)
gfa = pam.gfa
shm_coeff = pam.shm_coeff







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
