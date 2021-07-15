"""
============================================================================
Tracking with Robust Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD)
============================================================================

Here, we demonstrate fiber tracking using a probabalistic direction getter
and RUMBA-SD, a model introduced in [CanalesRodriguez2015]_. This model adapts
Richardson-Lucy deconvolution by assuming Rician or Noncentral Chi noise
instead of Gaussian, which more accurately reflects the noise from MRI
scanners (see also :ref:`example_reconst_rumba`). This tracking tutorial is an
extension on :ref:`example_tracking_probabilistic`.

We start by loading sample data and identifying a fiber response function.
"""

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, small_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.viz import window, actor, colormap, has_fury
from dipy.reconst.rumba import RumbaSD, global_fit

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
label_fname = get_fnames('stanford_labels')

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

seed_mask = (labels == 2)
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

sphere = small_sphere

"""
We can now initialize a `RumbaSD` model and fit it using the `global_fit`
method. For this example, TV regularization will be turned off for efficiency
although its usage can provide more coherent results in fiber tracking. The
fit will take about 15 minutes to complete.
"""

rumba = RumbaSD(gtab, lambda1=response[0][0], lambda2=response[0][1])
odf, f_iso, f_wm, combined = global_fit(rumba, data, sphere,
                                        mask=white_matter, use_tv=False)

"""
To establish stopping criterion, a common technique is to use the Generalized
Fractional Anisotropy (GFA). One point of caution is that RUMBA-SD by default
separates the fODF from an isotropic compartment. This can bias GFA results
computed on the fODF, although it will still generally be an effective
criterion.

However, an alternative stopping criterion that takes advantage of this
feature is to use RUMBA-SD's white matter volume fraction map.
"""

stopping_criterion = ThresholdStoppingCriterion(f_wm, .25)

"""
We can visualize a slice of this mask.
"""

import matplotlib.pyplot as plt

sli = f_wm.shape[2] // 2
plt.figure()

plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(f_wm[:, :, sli].T, cmap='gray', origin='lower')

plt.subplot(1, 2, 2).set_axis_off()
plt.imshow((f_wm[:, :, sli] > 0.25).T, cmap='gray', origin='lower')

plt.savefig('f_wm_tracking_mask.png')

"""
.. figure:: f_wm_trackin_mask.png
   :align: center

   White matter volume fraction slice
"""

"""
This discrete fODF can be used as a PMF in the `ProbabilisticDirectionGetter`
for sampling tracking directions. The PMF must be strictly non-negative;
RUMBA-SD already adheres to this constraint so no further manipulation of the
fODF is necessary.
"""

from dipy.direction import ProbabilisticDirectionGetter
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk

prob_dg = ProbabilisticDirectionGetter.from_pmf(odf, max_angle=30.,
                                                sphere=small_sphere)
streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                     affine, step_size=.5)
streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_probabilistic_dg_pmf.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(scene, out_path='tractogram_probabilistic_rumba.png',
                  size=(800, 800))
    if interactive:
        window.show(scene)

"""
.. figure:: tractogram_probabilistic_rumba.png
   :align: center

   RUMBA-SD tractogram
"""
