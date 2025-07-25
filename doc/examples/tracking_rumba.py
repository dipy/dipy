"""
============================================================================
Tracking with Robust Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD)
============================================================================

Here, we demonstrate fiber tracking using a probabilistic tracker
and RUMBA-SD, a model introduced in :footcite:p:`CanalesRodriguez2015`. This
model adapts Richardson-Lucy deconvolution by assuming Rician or Noncentral Chi
noise instead of Gaussian, which more accurately reflects the noise from MRI
scanners (see also
:ref:`sphx_glr_examples_built_reconstruction_reconst_rumba.py`). This tracking
tutorial is an extension on
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_probabilistic.py`.

We start by loading sample data and identifying a fiber response function.
"""

import matplotlib.pyplot as plt
from numpy.linalg import inv

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, small_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.rumba import RumbaSDModel
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking.tracker import probabilistic_tracking
from dipy.viz import actor, colormap, window

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")
t1_fname = get_fnames(name="stanford_t1")

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
t1_data, t1_aff = load_nifti(t1_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, affine, density=2)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

sphere = small_sphere

###############################################################################
# We can now initialize a `RumbaSdModel` model and fit it globally by setting
# `voxelwise` to `False`. For this example, TV regularization (`use_tv`) will
# be turned off for efficiency, although its usage can provide more coherent
# results in fiber tracking. The fit will take about 5 minutes to complete.

rumba = RumbaSDModel(
    gtab,
    wm_response=response[0],
    n_iter=200,
    voxelwise=False,
    use_tv=False,
    sphere=sphere,
)
rumba_fit = rumba.fit(data, mask=white_matter)
odf = rumba_fit.odf()  # fODF
f_wm = rumba_fit.f_wm  # white matter volume fractions

###############################################################################
# To establish stopping criterion, a common technique is to use the Generalized
# Fractional Anisotropy (GFA). One point of caution is that RUMBA-SD by default
# separates the fODF from an isotropic compartment. This can bias GFA results
# computed on the fODF, although it will still generally be an effective
# criterion.
#
# However, an alternative stopping criterion that takes advantage of this
# feature is to use RUMBA-SD's white matter volume fraction map.

stopping_criterion = ThresholdStoppingCriterion(f_wm, 0.25)

###############################################################################
# We can visualize a slice of this mask.

sli = f_wm.shape[2] // 2
plt.figure()

plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(f_wm[:, :, sli].T, cmap="gray", origin="lower")

plt.subplot(1, 2, 2).set_axis_off()
plt.imshow((f_wm[:, :, sli] > 0.25).T, cmap="gray", origin="lower")

plt.savefig("f_wm_tracking_mask.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# White matter volume fraction slice
#
#
#
# These discrete fODFs can be used as a PMF in the
# `ProbabilisticDirectionGetter` for sampling tracking directions. The PMF
# must be strictly non-negative; RUMBA-SD already adheres to this constraint
# so no further manipulation of the fODFs is necessary.

streamline_generator = probabilistic_tracking(
    seeds,
    stopping_criterion,
    affine,
    step_size=0.5,
    max_angle=30.0,
    sphere=sphere,
    sf=odf,
)
streamlines = Streamlines(streamline_generator)

color = colormap.line_colors(streamlines)
streamlines_actor = actor.streamtube(
    list(transform_streamlines(streamlines, inv(t1_aff))), colors=color, linewidth=0.1
)

vol_actor = actor.slicer(t1_data)
vol_actor.display(x=40)
vol_actor2 = vol_actor.copy()
vol_actor2.display(z=35)

scene = window.Scene()
scene.add(vol_actor)
scene.add(vol_actor2)
scene.add(streamlines_actor)
if interactive:
    window.show(scene)

window.record(
    scene=scene, out_path="tractogram_probabilistic_rumba.png", size=(800, 800)
)

sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_tractogram(sft, "tractogram_probabilistic_rumba.trx")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# RUMBA-SD tractogram
#
#
#
# References
# ----------
#
# .. footbibliography::
#
