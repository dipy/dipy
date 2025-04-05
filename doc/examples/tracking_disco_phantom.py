"""
=================================
Tractography on the DiSCo Phantom
=================================

This example compares probabilistic, deterministic and parallel
transport tractography (PTT) algorithms.

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

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
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
# Prepare the synthetic DiSCo data for tractography. The ground-truth
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
tracks_fname = fnames[disco1_fnames.index("DiSCo1_Strands_Trajectories.tck")]
GT_streams = load_tractogram(tracks_fname, reference=labels_img).streamlines
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
#
#

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
#
#

# Compute ODFs
data_fname = fnames[disco1_fnames.index("highRes_DiSCo1_DWI_RicianNoise-snr10.nii.gz")]
data = load_nifti_data(data_fname)

bvecs = fnames[disco1_fnames.index("DiSCo_gradients_dipy.bvecs")]
bvals = fnames[disco1_fnames.index("DiSCo_gradients.bvals")]
gtab = gradient_table(bvals=bvals, bvecs=bvecs)

response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
csd_fit = csd_model.fit(data, mask=mask)
ODFs = csd_fit.odf(default_sphere)

if has_fury:
    scene = window.Scene()
    ODF_spheres = actor.odf_slicer(
        ODFs[:, :, 17:18, :],
        sphere=default_sphere,
        scale=2,
        norm=False,
        colormap="plasma",
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
#
#

# Perform deterministic tractography using 1 thread (cpu)
print("Running Deterministic Tractography...")
streamline_generator = deterministic_tracking(
    seeds, sc, affine, sf=ODFs, nbr_threads=1, random_seed=42, sphere=default_sphere
)

det_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(det_streams, labels_img, Space.RASMM)
save_trk(sft, "tractogram_disco_deterministic.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(det_streams, colors=colormap.line_colors(det_streams)))
    window.record(
        scene=scene, out_path="tractogram_disco_deterministic.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(det_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (deterministic tractography): ", r)

plt.imshow(connectome, origin="lower", cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.savefig("connectome_deterministic.png")
plt.close()

###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Deterministic tractogram and corresponding connectome.
#
#
# Perform probabilistic tractography using 4 threads (cpus)

print("Running Probabilistic Tractography...")
streamline_generator = probabilistic_tracking(
    seeds, sc, affine, sf=ODFs, nbr_threads=4, random_seed=42, sphere=default_sphere
)
prob_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(prob_streams, labels_img, Space.RASMM)
save_trk(sft, "tractogram_disco_probabilistic.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(prob_streams, colors=colormap.line_colors(prob_streams)))
    window.record(
        scene=scene, out_path="tractogram_disco_probabilistic.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(prob_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (probabilistic tractography): ", r)

plt.imshow(connectome, origin="lower", cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.savefig("connectome_probabilistic.png")
plt.close()

###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo Probabilistic tractogram and corresponding connectome.
#
#

# Perform parallel transport tractography tractography using all threads (cpus)
print("Running Parallel Transport Tractography...")
streamline_generator = ptt_tracking(
    seeds,
    sc,
    affine,
    sf=ODFs,
    nbr_threads=0,
    random_seed=42,
    sphere=default_sphere,
    max_angle=20,
)
ptt_streams = Streamlines(streamline_generator)
sft = StatefulTractogram(ptt_streams, labels_img, Space.RASMM)
save_trk(sft, "tractogram_disco_ptt.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(ptt_streams, colors=colormap.line_colors(ptt_streams)))
    window.record(scene=scene, out_path="tractogram_disco_ptt.png", size=(800, 800))
    if interactive:
        window.show(scene)

# Compare the estimated connectivity with the ground-truth connectivity
connectome = connectivity_matrix(ptt_streams, affine, labels)[1:, 1:]
r, _ = pearsonr(
    GT_connectome[connectome_mask].flatten(), connectome[connectome_mask].flatten()
)
print("DiSCo ground-truth correlation (PTT tractography): ", r)
plt.imshow(connectome, origin="lower", cmap="viridis", interpolation="nearest")
plt.axis("off")
plt.savefig("connectome_ptt.png")
plt.close()

###############################################################################
#
# .. rst-class:: centered small fst-italic fw-semibold
#
# DiSCo PTT tractogram and corresponding connectome.
#
#
