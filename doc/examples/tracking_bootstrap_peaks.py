"""
====================================================
Bootstrap and Closest Peak Tracker Example
====================================================

This example shows how choices in tracker impact fiber
tracking results by demonstrating the bootstrap tracker (a type of
probabilistic tracking, as described in :footcite:p:`Berman2008` and the closest
peak tracker (a type of deterministic tracking)
:footcite:p:`Amirbekian2016`.

This example is an extension of the
:ref:`sphx_glr_examples_built_quick_start_tracking_introduction_eudx.py`
example. Let's start by loading the necessary modules for executing this
tutorial.
"""

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, small_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import bootstrap_tracking, closestpeak_tracking
from dipy.viz import actor, colormap, has_fury, window

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)


seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)

###############################################################################
# Next, we fit the CSD model.

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)

###############################################################################
# we use the CSA fit to calculate GFA, which will serve as our stopping
# criterion.

csa_model = CsaOdfModel(gtab, sh_order_max=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, 0.25)

###############################################################################
# Next, we need to set up our two trackers
#
#
# Example #1: Bootstrap tracker with CSD Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
boot_streamline_generator = bootstrap_tracking(
    seeds,
    stopping_criterion,
    affine,
    step_size=0.5,
    data=data,
    model=csd_model,
    max_angle=30.0,
    sphere=small_sphere,
)
streamlines = Streamlines(boot_streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_bootstrap_dg.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(scene=scene, out_path="tractogram_bootstrap_dg.png", size=(800, 800))
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum Bootstrap Probabilistic Tracker
#
#
# We have created a bootstrapped probabilistic set of streamlines. If you
# repeat the fiber tracking (keeping all inputs the same) you will NOT get
# exactly the same set of streamlines.
#
#
#
# Example #2: Closest peak tracker with CSD Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pmf = csd_fit.odf(small_sphere).clip(min=0)
peak_streamline_generator = closestpeak_tracking(
    seeds,
    stopping_criterion,
    affine,
    step_size=0.5,
    sf=pmf,
    max_angle=30.0,
    sphere=small_sphere,
)
streamlines = Streamlines(peak_streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "closest_peak_dg_CSD.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(
        scene=scene, out_path="tractogram_closest_peak_dg.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum Closest Peak Tracker
#
#
# We have created a set of streamlines using the closest peak tracker,
# which is a type of deterministic tracking. If you repeat the fiber tracking
# (keeping all inputs the same) you will get exactly the same set of
# streamlines.
#
#
#
# References
# ----------
#
# .. footbibliography::
#
