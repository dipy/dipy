"""
=============================================================
An introduction to the Deterministic Maximum Direction Getter
=============================================================

Deterministic maximum direction getter is the deterministic version of the
probabilistic direction getter. It can be used with the same local models
and has the same parameters. Deterministic maximum fiber tracking follows
the trajectory of the most probable pathway within the tracking constraint
(e.g. max angle). In other words, it follows the direction with the highest
probability from a distribution, as opposed to the probabilistic direction
getter which draws the direction from the distribution. Therefore, the maximum
deterministic direction getter is equivalent to the probabilistic direction
getter returning always the maximum value of the distribution.

Deterministic maximum fiber tracking is an alternative to EuDX deterministic
tractography and unlike EuDX does not follow the peaks of the local models but
uses the entire orientation distributions.

This example is an extension of the
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_probabilistic.py`
example. We begin by loading the data, fitting a Constrained Spherical
Deconvolution (CSD) reconstruction model for the tractography and fitting
the constant solid angle (CSA) reconstruction model to define the tracking
mask (stopping criterion).
"""

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor, colormap, has_fury

# Enables/disables interactive visualization
interactive = False


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
label_fname = get_fnames('stanford_labels')

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)

csa_model = CsaOdfModel(gtab, sh_order_max=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)

###############################################################################
# The Fiber Orientation Distribution (FOD) of the CSD model estimates the
# distribution of small fiber bundles within each voxel. This distribution
# can be used for deterministic fiber tracking. As for probabilistic tracking,
# there are many ways to provide those distributions to the deterministic
# maximum direction getter. Here, the spherical harmonic representation of
# the FOD is used.

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    csd_fit.shm_coeff, max_angle=30., sphere=default_sphere, sh_to_pmf=True)
streamline_generator = LocalTracking(detmax_dg, stopping_criterion, seeds,
                                     affine, step_size=.5)
streamlines = Streamlines(streamline_generator)

sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_deterministic_dg.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(scene, out_path='tractogram_deterministic_dg.png',
                  size=(800, 800))
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using deterministic maximum direction getter
