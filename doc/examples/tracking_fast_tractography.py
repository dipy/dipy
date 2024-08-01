"""
===========================================
An introduction to the Fast Tracking Module
===========================================

The fast tracking module allow to run tractography on multiple CPU cores.

Current implemented algorithms are probabilistic, deterministic and parallel
transport tractography (PTT).

See
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_probabilistic.py`
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_deterministic.py`
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_ptt.py`
for detailed examples of those algorithms.
"""

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

from dipy.core.sphere import HemiSphere
from dipy.data import get_fnames, get_sphere
from dipy.direction.peaks import peaks_from_positions
from dipy.direction.pmf import SimplePmfGen
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_trk
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
# Prepare the synthetic DiSCo data for fast tracking. The ground-truth
# connectome will be use to evaluate tractography performances.
fnames = get_fnames("disco1")
GT_connectome = np.loadtxt(fnames[35])
connectome_mask = np.tril(np.ones(GT_connectome.shape), -1) > 0
labels_img = nib.load(fnames[23])
labels = np.round(labels_img.get_fdata()).astype(int)
GT_streams = load_tractogram(fnames[39], reference=labels_img).streamlines
if has_fury:
    scene = window.Scene()
    scene.add(actor.line(GT_streams, colors=colormap.line_colors(GT_streams)))
    window.record(
        scene=scene, out_path="tractogram_ground_truth.png", size=(800, 800))
    if interactive:
        window.show(scene)

plt.imshow(GT_connectome, origin="lower",
           cmap="viridis", interpolation="nearest")
plt.axis('off')
plt.savefig("connectome_ground_truth.png")
plt.close()
###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo ground-truth trajectories (left) and connectivity matrix (right).

###############################################################################
# Prepare ODFs
sphere = get_sphere("repulsion724")
GT_SH_img = nib.load(fnames[20])
GT_SH = GT_SH_img.get_fdata()
GT_ODF = sh_to_sf(GT_SH, sphere, sh_order_max=12,
                  basis_type="tournier07", legacy=False)
GT_ODF[GT_ODF < 0] = 0
pmf_gen = SimplePmfGen(np.asarray(GT_ODF, dtype=float), sphere)

if has_fury:
    scene = window.Scene()
    ODF_spheres = actor.odf_slicer(GT_ODF[:, :, 17:18, :], sphere=sphere,
                                   scale=2, norm=False, colormap="plasma")
    scene.add(ODF_spheres)
    window.record(scene=scene, out_path="GT_odfs.png", size=(600, 600))
    if interactive:
        window.show(scene)
###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo ground-truth ODFs.

###############################################################################
# Tracking mask, seed positions and initial directions
mask = nib.load(fnames[25]).get_fdata()
sc = BinaryStoppingCriterion(mask)

affine = nib.load(fnames[25]).affine
voxel_size = np.ones(3)
seed_mask = nib.load(fnames[34]).get_fdata()
seed_mask = binary_erosion(seed_mask * mask, iterations=1)
seeds_positions = random_seeds_from_mask(seed_mask,
                                         affine,
                                         seeds_count=5000,
                                         seed_count_per_voxel=False)

peaks = peaks_from_positions(
    seeds_positions, GT_ODF, sphere, npeaks=1, affine=affine)
seeds, initial_directions = seeds_directions_pairs(
    seeds_positions, peaks, max_cross=1)

plt.imshow(seed_mask[:, :, 17], origin="lower",
           cmap="gray", interpolation="nearest")
plt.axis('off')
plt.title('Seeding Mask')
plt.savefig("seeding_mask.png")
plt.close()
plt.imshow(mask[:, :, 17], origin="lower",
           cmap="gray", interpolation="nearest")
plt.axis('off')
plt.title('Tracking Mask')
plt.savefig("tracking_mask.png")
plt.close()
###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo seeding (left) and tracking (right) masks.

###############################################################################
# Perform fast deterministic tractography
det_params = generate_tracking_parameters("det",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=voxel_size,
                                          max_angle=20
                                          )

streamline_generator = generate_tractogram(seeds,
                                           initial_directions,
                                           sc,
                                           det_params,
                                           pmf_gen
                                           )
det_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(det_streams, GT_SH_img, Space.RASMM)
save_trk(sft, "tractogram_fast_deterministic.trk")

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(det_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(GT_connectome[connectome_mask].flatten(),
                connectome[connectome_mask].flatten())
print("DiSCo ground-truth correlation (fast deterministic tractography): ", r)

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(det_streams, colors=colormap.line_colors(det_streams)))
    window.record(
        scene=scene, out_path="tractogram_fast_deterministic.png", size=(800, 800))
    if interactive:
        window.show(scene)

plt.imshow(connectome, origin="lower",
           cmap="viridis", interpolation="nearest")
plt.axis('off')
plt.savefig("connectome_deterministic.png")
plt.close()
###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Deterministic tractogram and corresponding connectome.

###############################################################################
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

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(prob_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(GT_connectome[connectome_mask].flatten(),
                connectome[connectome_mask].flatten())
print("DiSCo ground-truth correlation (fast probabilistic tractography): ", r)

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(prob_streams, colors=colormap.line_colors(prob_streams)))
    window.record(
        scene=scene, out_path="tractogram_fast_probabilistic.png", size=(800, 800))
    if interactive:
        window.show(scene)

plt.imshow(connectome, origin="lower",
           cmap="viridis", interpolation="nearest")
plt.axis('off')
plt.savefig("connectome_probabilistic.png")
plt.close()
###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Probabilistic tractogram and corresponding connectome.

###############################################################################
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

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(ptt_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(GT_connectome[connectome_mask].flatten(),
                connectome[connectome_mask].flatten())
print("DiSCo ground-truth correlation (fast PTT tractography): ", r)

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(ptt_streams, colors=colormap.line_colors(ptt_streams)))
    window.record(
        scene=scene, out_path="tractogram_fast_ptt.png", size=(800, 800))
    if interactive:
        window.show(scene)

plt.imshow(connectome, origin="lower",
           cmap="viridis", interpolation="nearest")
plt.axis('off')
plt.savefig("connectome_ptt.png")
plt.close()
###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo PTT tractogram and corresponding connectome.

###############################################################################
