"""
=====================================================
Metal and GPU probabilistic tracking (simple tracker)
=====================================================

DIPY's "simple" tracker :footcite:p:`Kruper2025` is a stripped-down
probabilistic tracker that runs on the GPU through one of three backends:
CUDA (NVIDIA), Metal (Apple Silicon) or WebGPU (any GPU with subgroup
support). It generates tens of millions of streamlines in minutes, at the
price of a few restrictions compared with
:func:`dipy.tracking.tracker.probabilistic_tracking` on the CPU:

- the whole PMF (spherical function) must fit in memory, so the ODF is
  sampled on a sphere rather than kept as spherical harmonics;
- voxels must be isotropic;
- the stopping criterion must be a
  :class:`dipy.tracking.stopping_criterion.ThresholdStoppingCriterion`;
- the streamline length is capped at a fixed number of steps.

The GPU dependencies are optional. Install the one matching your hardware::

    pip install "dipy[cu12]"     # or dipy[cu13], for CUDA 12 / 13
    pip install "dipy[metal]"    # Apple Silicon
    pip install "dipy[webgpu]"   # cross-platform, needs a wgpu adapter

This example follows :ref:`sphx_glr_examples_built_fiber_tracking_tracking_probabilistic.py`
and only changes the tracking call and number of seeds. We start by fitting a CSD model.
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram, save_trx_from_generator
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.dti import TensorModel
from dipy.tracking.simplet import SIMPLE_BACKENDS, simple_backend_available
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import probabilistic_tracking
from dipy.tracking.utils import random_seeds_from_mask
from dipy.viz import has_fury

if has_fury:
    from fury import actor, colormap, window

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

white_matter = (labels == 1) | (labels == 2)
seed_mask = labels == 2
seeds = random_seeds_from_mask(
    seed_mask, affine, seeds_count=20, seed_count_per_voxel=True, random_seed=1
)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)

###############################################################################
# The simple tracker stops on a scalar map. Here we use the FA from a tensor
# fit, thresholded at 0.2.

tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(data, mask=white_matter)
stopping_criterion = ThresholdStoppingCriterion(tensor_fit.fa, 0.2)

###############################################################################
# Pick the first GPU backend available on this machine, falling back to the
# CPU when there is none (e.g. when this example is built on a server without
# a GPU). ``backend="auto"`` does the same, but raises if no GPU backend is
# available.

backend = "cpu"
for name in SIMPLE_BACKENDS:
    if simple_backend_available(name):
        backend = name
        break
print(f"Tracking with the {backend} backend")

###############################################################################
# The tracking call is the same as on the CPU, plus the ``backend`` argument.
# Passing ``sh`` is fine: the coefficients are sampled on ``sphere`` before
# being uploaded to the GPU.

streamline_generator = probabilistic_tracking(
    seeds,
    stopping_criterion,
    affine,
    sh=csd_fit.shm_coeff,
    sphere=default_sphere,
    random_seed=1,
    max_angle=20,
    step_size=0.5,
    backend=backend,
)

streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_tractogram(sft, "tractogram_simplet.trx")

if has_fury:
    colors = np.repeat(
        colormap.line_colors(streamlines), [len(sl) for sl in streamlines], axis=0
    )
    lines_actor = actor.streamlines(lines=streamlines, colors=colors)
    window.snapshot(actors=[lines_actor], fname="tractogram_simplet.png")
    if interactive:
        window.show([lines_actor])

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Using TRX with the simple probabilistic tracker
#
#
#
# The GPU produces streamlines much faster than they can be collected one at
# a time in Python. With ``chunked=True`` the generator yields one
# ``(points, lengths)`` tuple per chunk of ``chunk_size`` seeds instead, and
# :func:`dipy.io.streamline.save_trx_from_generator` writes those chunks
# straight to a TRX file, so whole-brain tractograms of hundreds of
# gigabytes never have to fit in memory.

streamline_generator = probabilistic_tracking(
    seeds,
    stopping_criterion,
    affine,
    sh=csd_fit.shm_coeff,
    sphere=default_sphere,
    random_seed=1,
    max_angle=20,
    step_size=0.5,
    backend=backend,
    chunked=True,
    chunk_size=25000,
)

trx = save_trx_from_generator(
    streamline_generator,
    hardi_img,
    filename="tractogram_simplet_chunked.trx",
    nb_streamlines_estimate=len(seeds),
)
print(f"Saved {trx.header['NB_STREAMLINES']} streamlines")

###############################################################################
# The ``dipy_track`` command line tool uses the same chunked writer whenever
# the output tractogram is a ``.trx`` file.
#
#
#
# References
# ----------
#
# .. footbibliography::
#
