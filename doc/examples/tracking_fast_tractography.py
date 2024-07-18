"""
===========================================
An introduction to the Fast Tracking Module
===========================================

The fast tracking module allow to run tractography on multiple CPU cores.

Current Implemented algorithms are Probabilistic, Deterministic and Parallel
Transport Tractography (PTT).

See
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_probabilistic.py`
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_deterministic.py`
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_ptt.py`
for detailed examples of those algorithms.
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

from dipy.core.sphere import HemiSphere
from dipy.data import get_fnames, get_sphere
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import SimplePmfGen
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.shm import sh_to_sf
from dipy.tracking.fast_tracking import generate_tractogram
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.utils import (connectivity_matrix, random_seeds_from_mask,
                                 seeds_directions_pairs)
from dipy.viz import actor, colormap, has_fury, window

# Enables/disables interactive visualization
interactive = False

###############################################################################
# Prepare the synthetic DiSCo data for fast tracking
fnames = get_fnames("disco1")

# prepare the GT connectome data
GT_connectome = np.loadtxt(fnames[35])
connectome_mask = np.tril(np.ones(GT_connectome.shape), -1) > 0
labels = np.round(nib.load(fnames[23]).get_fdata()).astype(int)

# prepare ODFs
sphere = HemiSphere.from_sphere(get_sphere("repulsion724"))
GT_SH_img = nib.load(fnames[20])
GT_SH = GT_SH_img.get_fdata()
GT_ODF = sh_to_sf(GT_SH, sphere, sh_order_max=12,
                  basis_type="tournier07", legacy=False)
GT_ODF[GT_ODF < 0] = 0

# seeds position and initial directions
mask = nib.load(fnames[25]).get_fdata()
affine = nib.load(fnames[25]).affine
voxel_size = np.ones(3)
seed_mask = nib.load(fnames[34]).get_fdata()
seed_mask = binary_erosion(seed_mask * mask, iterations=1)
seeds_positions = random_seeds_from_mask(seed_mask,
                                         affine,
                                         seeds_count=5000,
                                         seed_count_per_voxel=False
                                         )

pmf_gen = SimplePmfGen(np.asarray(GT_ODF, dtype=float), sphere)
peaks = peaks_from_positions(
    seeds_positions, GT_ODF, sphere, npeaks=1, affine=affine)
seeds, initial_directions = seeds_directions_pairs(
    seeds_positions, peaks, max_cross=1)

# stopping criterion
sc = BinaryStoppingCriterion(mask)

###############################################################################
# Perform fast deterministic tractography
det_params = generate_tracking_parameters("det",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=voxel_size,
                                          max_angle=20
                                          )

# Prepare the streamline generator
streamline_generator = generate_tractogram(seeds,
                                           initial_directions,
                                           sc,
                                           det_params,
                                           pmf_gen
                                           )
det_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(det_streams, GT_SH_img, Space.RASMM)
save_trk(sft, "tractogram_fast_deterministic.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(det_streams, colormap.line_colors(det_streams)))
    window.record(
        scene, out_path="tractogram_fast_deterministic.png", size=(800, 800))
    if interactive:
        window.show(scene)

# Perform fast probabilistic tractography
prob_params = generate_tracking_parameters("prob",
                                           max_len=500,
                                           step_size=0.2,
                                           voxel_size=voxel_size,
                                           max_angle=20
                                           )

# Prepare the streamline generator
streamline_generator = generate_tractogram(seeds,
                                           initial_directions,
                                           sc,
                                           prob_params,
                                           pmf_gen
                                           )
prob_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(prob_streams, GT_SH_img, Space.RASMM)
save_trk(sft, "tractogram_fast_probabilistic.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(prob_streams, colormap.line_colors(prob_streams)))
    window.record(
        scene, out_path="tractogram_fast_probabilistic.png", size=(800, 800))
    if interactive:
        window.show(scene)

# Perform fast paralle transport tractography tractography
ptt_params = generate_tracking_parameters("ptt",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=voxel_size,
                                          max_angle=15,
                                          probe_quality=4
                                          )

# Prepare the streamline generator
streamline_generator = generate_tractogram(seeds,
                                           initial_directions,
                                           sc,
                                           ptt_params,
                                           pmf_gen
                                           )
ptt_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(ptt_streams, GT_SH_img, Space.RASMM)
save_trk(sft, "tractogram_fast_ptt.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(ptt_streams, colormap.line_colors(ptt_streams)))
    window.record(
        scene, out_path="tractogram_fast_ptt.png", size=(800, 800))
    if interactive:
        window.show(scene)

###############################################################################
# .. _tractogram_fast_deterministic:
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Deterministic tractogram.
#
#
# .. _tractogram_fast_probabilistic:
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Probabilistic tractogram.
# .. _tractogram_fast_probabilistic:
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Probabilistic tractogram.
#
#
# .. _tractogram_fast_ptt:
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo PTT tractogram.
#
# The fast tractography result on the DiSCo dataset.
#


###############################################################################
# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(det_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(
    ), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (fast deterministic tractography): ", r)

connectome = connectivity_matrix(prob_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(
    ), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (fast probabilistc tractography): ", r)

connectome = connectivity_matrix(ptt_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(
    ), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (fast PTT tractography): ", r)
###############################################################################
# Prepare the brain data for fast tracking
