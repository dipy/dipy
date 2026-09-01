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
from dipy.segment.mask import median_otsu

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
# Load the dMRI data.

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

denoised = mppca(data, mask=mask, patch_radius=2)

axial_slice = 10
plt.imshow(np.rot90(denoised[:, :, axial_slice, 0]), cmap=plt.cm.bone)
plt.imshow(np.rot90(data[:, :, axial_slice, 0]), cmap=plt.cm.hot,
           alpha=0.25)
plt.show()

qball_model = shm.QballModel(gtab, 8)

peaks = dp.peaks_from_model(model=qball_model, data=denoised,
                            relative_peak_threshold=0.5,
                            min_separation_angle=25,
                            sphere=sphere, mask=mask)

ap = shm.anisotropic_power(peaks.shm_coeff)

plt.imshow(np.rot90(ap[:, :, axial_slice]), cmap=plt.cm.bone)
plt.show()

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(
    ap, nclasses=3, beta=0.1)

csf = np.where(final_segmentation == 1, 1, 0)
gm = np.where(final_segmentation == 2, 1, 0)
wm = np.where(final_segmentation == 3, 1, 0)

plt.imshow(np.rot90(denoised[:, :, axial_slice, 0]), cmap=plt.cm.bone)
plt.imshow(np.rot90(np.where(gm[:, :, axial_slice], 0.5, np.nan)),
           cmap=plt.cm.bone_r, vmin=0, vmax=1, alpha=0.5)
plt.imshow(np.rot90(np.where(wm[:, :, axial_slice], 0.9, np.nan)),
           cmap=plt.cm.bone, vmin=0, vmax=1, alpha=0.5)
plt.imshow(np.rot90(np.where(csf[:, :, axial_slice], 1, np.nan)),
           cmap=plt.cm.cool, alpha=0.5)
plt.show()

mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data, roi_radii=10,
                                                    wm_fa_thr=0.7,
                                                    gm_fa_thr=0.3,
                                                    csf_fa_thr=0.15,
                                                    gm_md_thr=0.001,
                                                    csf_md_thr=0.0032)

###############################################################################
# Now, generate a response function from the data.

response_wm, response_gm, response_csf = response_from_mask_msmt(
    gtab, data, wm, gm, csf)

ubvals = unique_bvals_tolerance(gtab.bvals)
response_mcsd = multi_shell_fiber_response(sh_order_max=8,
                                           bvals=ubvals,
                                           wm_rf=response_wm,
                                           gm_rf=response_gm,
                                           csf_rf=response_csf)

mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
mcsd_fit = mcsd_model.fit(denoised)

mcsd_pred = mcsd_fit.predict()
mcsd_odf = mcsd_fit.odf(sphere)

###############################################################################
# track fibers
stopping_criterion = ThresholdStoppingCriterion(gfa, 0.25)
dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    peaks.shm_coeff, max_angle=30., sphere=sphere, sh_to_pmf=True)
seeds = seeds_from_mask(wm_mask, affine, density=(1, 1, 1))
streamline_generator = LocalTracking(
    dg, stopping_criterion, seeds, affine, step_size=0.5)
streamlines = Streamlines(streamline_generator)

sft = StatefulTractogram(streamlines, dwi, Space.RASMM)
os.makedirs(op.join(out_dir, f'sub-{sub}', 'det'), exist_ok=True)
save_trk(sft, op.join(out_dir, f'sub-{sub}', 'det', 'tractogram.trk'))

