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

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

from dipy.data import get_fnames, get_sphere
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_trk
from dipy.reconst.shm import sh_to_sf
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import (
    deterministic_tracking,
    probabilistic_tracking,
    ptt_tracking,
)
from dipy.tracking.utils import connectivity_matrix, seeds_from_mask
from dipy.viz import actor, colormap, has_fury, window

# Enables/disables interactive visualization
interactive = False

print("Downloading data...")
###############################################################################
# Prepare the synthetic DiSCo data for fast tracking. The ground-truth
# connectome will be use to evaluate tractography performances.
fnames = get_fnames(name="disco1")
disco1_fnames = [os.path.basename(f) for f in fnames]

GT_connectome_fname = fnames[
    disco1_fnames.index("DiSCo1_Connectivity_Matrix_Cross-Sectional_Area.txt")
]
GT_connectome = np.loadtxt(GT_connectome_fname)

# select the non-zero connections of the upper triangular part of the connectome
connectome_mask = np.tril(np.ones(GT_connectome.shape), -1) > 0

# load the
labels_fname = fnames[disco1_fnames.index("highRes_DiSCo1_ROIs.nii.gz")]
labels, affine, labels_img = load_nifti(labels_fname, return_img=True)
labels = labels.astype(int)

print("Loading data...")

GT_streams = load_tractogram(fnames[39], reference=labels_img).streamlines
if has_fury:
    scene = window.Scene()
    scene.add(actor.line(GT_streams, colors=colormap.line_colors(GT_streams)))
    window.record(scene=scene, out_path="tractogram_ground_truth.png", size=(800, 800))
    if interactive:
        window.show(scene)

plt.imshow(GT_connectome, origin="lower", cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.savefig("connectome_ground_truth.png")
plt.close()

###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo ground-truth trajectories (left) and connectivity matrix (right).

###############################################################################
# Prepare ODFs
sphere = get_sphere(name="repulsion724")

GT_SH_fname = fnames[disco1_fnames.index("highRes_DiSCo1_Strand_ODFs.nii.gz")]
GT_SH = load_nifti_data(GT_SH_fname)
GT_ODF = sh_to_sf(GT_SH, sphere, sh_order_max=12, basis_type="tournier07", legacy=False)
GT_ODF[GT_ODF < 0] = 0

if has_fury:
    scene = window.Scene()
    ODF_spheres = actor.odf_slicer(
        GT_ODF[:, :, 17:18, :], sphere=sphere, scale=2, norm=False, colormap="plasma"
    )
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
mask_fname = fnames[disco1_fnames.index("highRes_DiSCo1_mask.nii.gz")]
mask = load_nifti_data(mask_fname)
sc = BinaryStoppingCriterion(mask)

voxel_size = np.ones(3)
seed_fname = fnames[disco1_fnames.index("highRes_DiSCo1_ROIs-mask.nii.gz")]
seed_mask = load_nifti_data(seed_fname)
seed_mask = binary_erosion(seed_mask * mask, iterations=1)
seeds = seeds_from_mask(seed_mask, affine, density=2)

plt.imshow(seed_mask[:, :, 17], origin="lower", cmap="gray", interpolation="nearest")
plt.axis("off")
plt.title("Seeding Mask")
plt.savefig("seeding_mask.png")
plt.close()
plt.imshow(mask[:, :, 17], origin="lower", cmap="gray", interpolation="nearest")
plt.axis("off")
plt.title("Tracking Mask")
plt.savefig("tracking_mask.png")
plt.close()
###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo seeding (left) and tracking (right) masks.

###############################################################################
# Perform fast deterministic tractography using 1 thread (cpu)

print("Running fast Deterministic Tractography...")
streamline_generator = deterministic_tracking(
    seeds,
    sc,
    affine,
    sf=GT_ODF,
    nbr_threads=1,
    random_seed=1,
    sphere=sphere,
)

det_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(det_streams, labels_img, Space.RASMM)
save_trk(sft, "tractogram_fast_deterministic.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(det_streams, colors=colormap.line_colors(det_streams)))
    window.record(
        scene=scene, out_path="tractogram_fast_deterministic.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(det_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (fast deterministic tractography): ", r)

plt.imshow(connectome, origin="lower", cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.savefig("connectome_deterministic.png")
plt.close()

###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Deterministic tractogram and corresponding connectome.

###############################################################################
# Perform fast probabilistic tractography using 4 threads (cpus)

print("Running fast Probabilistic Tractography...")
streamline_generator = probabilistic_tracking(
    seeds,
    sc,
    affine,
    sf=GT_ODF,
    nbr_threads=4,
    random_seed=1,
    sphere=sphere,
)
prob_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(prob_streams, labels_img, Space.RASMM)
save_trk(sft, "tractogram_fast_probabilistic.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(prob_streams, colors=colormap.line_colors(prob_streams)))
    window.record(
        scene=scene, out_path="tractogram_fast_probabilistic.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(prob_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (fast probabilistic tractography): ", r)

plt.imshow(connectome, origin="lower", cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.savefig("connectome_probabilistic.png")
plt.close()

###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Probabilistic tractogram and corresponding connectome.

###############################################################################

# Perform fast parallel transport tractography tractography using all threads (cpus)
print("Running fast Parallel Transport Tractography...")
streamline_generator = ptt_tracking(
    seeds,
    sc,
    affine,
    sf=GT_ODF,
    nbr_threads=0,
    random_seed=1,
    sphere=sphere,
)
ptt_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(ptt_streams, labels_img, Space.RASMM)
save_trk(sft, "tractogram_fast_ptt.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(ptt_streams, colors=colormap.line_colors(ptt_streams)))
    window.record(scene=scene, out_path="tractogram_fast_ptt.png", size=(800, 800))
    if interactive:
        window.show(scene)

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(ptt_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (fast PTT tractography): ", r)
plt.imshow(connectome, origin="lower", cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.savefig("connectome_ptt.png")
plt.close()

###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo PTT tractogram and corresponding connectome.

###############################################################################
