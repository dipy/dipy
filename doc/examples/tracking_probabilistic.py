"""
=================================================
An introduction to the Probabilistic Tractography
=================================================

Probabilistic fiber tracking is a way of reconstructing white matter
connections using diffusion MR imaging. Like deterministic fiber tracking, the
probabilistic approach follows the trajectory of a possible pathway step by
step starting at a seed, however, unlike deterministic tracking, the tracking
direction at each point along the path is chosen at random from a distribution.
The distribution at each point is different and depends on the observed
diffusion data at that point. The distribution of tracking directions at each
point can be represented as a probability mass function (PMF) if the possible
tracking directions are restricted to discrete numbers of well distributed
points on a sphere.

This example is an extension of the
:ref:`sphx_glr_examples_built_quick_start_tracking_introduction_eudx.py`
example. We'll begin by repeating a few steps from that example, loading the
data and fitting a Constrained Spherical Deconvolution (CSD) model.
"""

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import probabilistic_tracking
from dipy.tracking.utils import seeds_from_mask
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
seeds = seeds_from_mask(seed_mask, affine, density=2)

white_matter = (labels == 1) | (labels == 2)
sc = BinaryStoppingCriterion(white_matter)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)


###############################################################################
# The Fiber Orientation Distribution (FOD) of the CSD model estimates the
# distribution of small fiber bundles within each voxel. We can use this
# distribution for probabilistic fiber tracking. One way to do this is to
# represent the FOD using a discrete sphere. This discrete FOD can be used by
# ``probabilistic_tracking`` as a PMF (sf or spherical function) for sampling
# tracking directions.

fod = csd_fit.odf(default_sphere)
streamline_generator = probabilistic_tracking(
    seeds,
    sc,
    affine,
    sf=fod,
    random_seed=1,
    sphere=default_sphere,
    max_angle=20,
    step_size=0.2,
)

streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_probabilistic_sf.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(
        scene=scene, out_path="tractogram_probabilistic_sf.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using probabilistic tractography from PMF
#
#
#
# One disadvantage of using a discrete PMF to represent possible tracking
# directions is that it tends to take up a lot of memory (RAM). The size of the
# PMF, the FOD in this case, must be equal to the number of possible tracking
# directions on the hemisphere, and every voxel has a unique PMF. In this case
# the data is ``(81, 106, 76)`` and ``small_sphere`` has 181 directions so the
# FOD is ``(81, 106, 76, 181)``. One way to avoid sampling the PMF and holding
# it in memory is to use directly from the spherical
# harmonic (SH) representation of the FOD. By using this approach, we can also
# use a larger sphere, like ``default_sphere`` which has 362 directions on the
# hemisphere, without having to worry about memory limitations.

streamline_generator = probabilistic_tracking(
    seeds,
    sc,
    affine,
    sh=csd_fit.shm_coeff,
    random_seed=1,
    sphere=default_sphere,
    max_angle=20,
    step_size=0.2,
)

streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_probabilistic_sh.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(
        scene=scene, out_path="tractogram_probabilistic_sh.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using probabilistic tractography from SH
#
#
#
# Not all model fits have the ``shm_coeff`` attribute because not all models
# use this basis to represent the data internally. However we can fit the ODF
# of any model to the spherical harmonic basis using the ``peaks_from_model``
# function.

peaks = peaks_from_model(
    csd_model,
    data,
    default_sphere,
    0.5,
    25,
    mask=white_matter,
    return_sh=True,
    parallel=True,
    num_processes=1,
)
fod_coeff = peaks.shm_coeff

streamline_generator = probabilistic_tracking(
    seeds,
    sc,
    affine,
    sh=fod_coeff,
    random_seed=1,
    sphere=default_sphere,
    max_angle=20,
    step_size=0.2,
)

streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_probabilistic_sh_pfm.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(
        scene=scene, out_path="tractogram_probabilistic_sh_pfm.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using probabilistic tracker from SH
# (peaks_from_model)
