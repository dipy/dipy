"""
==============================
Introduction to Basic Tracking
==============================

Local fiber tracking is an approach used to model white matter fibers by
creating streamlines from local directional information. The idea is as
follows: if the local directionality of a tract/pathway segment is known, one
can integrate along those directions to build a complete representation of that
structure. Local fiber tracking is widely used in the field of diffusion MRI
because it is simple and robust.

In order to perform local fiber tracking, three things are needed:

1. A method for getting directions from a diffusion dataset.
2. A method for identifying when the tracking must stop.
3. A set of seeds from which to begin tracking.

This example shows how to combine the 3 parts described above
to create a tractography reconstruction from a diffusion data set using
the EuDX algorithm :footcite:p:`Garyfallidis2012b`.

EuDX is a fast deterministic tracking algorithm that operates directly on
discrete peaks extracted from the diffusion data, rather than on continuous
orientation distribution functions (ODFs). This makes it computationally
efficient and well-suited for quick exploratory tractography. A key feature
of EuDX is its ability to handle fiber crossings by tracking along multiple
peak directions at each seed point.

Let's begin by importing the necessary modules.
"""

import matplotlib.pyplot as plt
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst.force import FORCEModel, force_peaks
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import eudx_tracking
from dipy.viz import actor, colormap, has_fury, window

###############################################################################
# Now, let's load an HARDI dataset from Stanford. If you have
# not already downloaded this data set, the first time you run this example you
# will need to be connected to the internet and this dataset will be downloaded
# to your computer.

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

###############################################################################
# This dataset provides a label map in which all white matter tissues are
# labeled either 1 or 2. Let's create a white matter mask to restrict tracking
# to the white matter.

white_matter = (labels == 1) | (labels == 2)

###############################################################################
# Step 1: Getting directions from a diffusion dataset
# ---------------------------------------------------
#
# The first thing we need to begin fiber tracking is a way of getting
# directions from this diffusion data set. Here, we use the FORCE model
# :footcite:p:`Shah2025` to estimate fiber orientations. FORCE is a
# forward-modeling approach that simulates a library of biologically
# plausible fiber configurations and matches each voxel's signal to its
# nearest library entry. We then extract discrete peaks from the FORCE fit
# using ``force_peaks``, which produces a PeaksAndMetrics (PAM) object
# containing peak directions, peak values, and spherical harmonic
# coefficients at each voxel. EuDX tracking operates directly on these
# extracted peaks.

_, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1)

model = FORCEModel(gtab, n_neighbors=50, use_posterior=False, verbose=True)
model.generate(num_simulations=500000, num_cpus=-1, verbose=True, use_cache=True)
fit = model.fit(data, mask=mask, n_jobs=-1, verbose=True)

force_pam = force_peaks(fit)

###############################################################################
# For quality assurance we can also visualize a slice from the direction field
# which we will use as the basis to perform the tracking. The visualization
# will be done using the ``fury`` python package

if has_fury:
    scene = window.Scene()
    scene.add(
        actor.peak_slicer(
            force_pam.peak_dirs, peaks_values=force_pam.peak_values, colors=None
        )
    )

    window.record(scene=scene, out_path="force_direction_field.png", size=(900, 900))

    if interactive:
        window.show(scene, size=(800, 800))

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Direction Field (peaks)
#
#
# Step 2: Identifying when the tracking must stop
# -----------------------------------------------
# Next we need some way of restricting the fiber tracking to areas with good
# directionality information. We've already created the white matter mask,
# but we can go a step further and restrict fiber tracking to those areas where
# the FORCE model shows significant restricted diffusion by thresholding on
# the fractional anisotropy (FA) estimated by FORCE.

stopping_criterion = ThresholdStoppingCriterion(fit.fa, 0.25)

###############################################################################
# Again, for quality assurance, we can also visualize a slice of the FA and
# the resulting tracking mask.

sli = fit.fa.shape[2] // 2
plt.figure("FA")
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(np.rot90(fit.fa[:, :, sli]), cmap="gray")

plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(np.rot90((fit.fa[:, :, sli] > 0.25).astype(float)), cmap="gray")

plt.savefig("fa_tracking_mask.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# An example of a tracking mask derived from the fractional
# anisotropy (FA).
#
#
#
# Step 3: Defining a set of seeds from which to begin tracking
# ------------------------------------------------------------
# Before we can begin tracking, we need to specify where to "seed" (begin)
# the fiber tracking. Generally, the seeds chosen will depend on the pathways
# one is interested in modeling. In this example, we'll use a
# $2 \times 2 \times 2$ grid of seeds per voxel, in a sagittal slice of the
# corpus callosum. Tracking from this region will give us a model of the
# corpus callosum tract. This slice has label value ``2`` in the label's image.

seed_mask = labels == 2
seeds = utils.seeds_from_mask(seed_mask, affine, density=[2, 2, 2])

###############################################################################
# Tracking
# --------
# Finally, we can bring it all together using ``eudx_tracking``. The EuDX
# algorithm :footcite:p:`Garyfallidis2012b` is a fast deterministic tracking
# method that follows the peaks extracted from the diffusion data. Unlike
# other tracking methods such as deterministic or probabilistic tracking that
# operate on continuous orientation distributions (SH or SF), EuDX works
# directly with the discrete peak directions stored in the PAM object.

streamline_generator = eudx_tracking(
    seeds, stopping_criterion, affine, pam=force_pam, step_size=0.5, random_seed=1
)
streamlines = Streamlines(streamline_generator)

###############################################################################
# We will then display the resulting streamlines using the ``fury``
# python package.

if has_fury:
    # Prepare the display objects.
    streamlines_actor = actor.line(
        streamlines, colors=colormap.line_colors(streamlines)
    )

    # Create the 3D display.
    scene = window.Scene()
    scene.add(streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    window.record(scene=scene, out_path="tractogram_EuDX.png", size=(800, 800))
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using EuDx
#
#
# We've created a deterministic set of streamlines using the EuDX algorithm.
# This is so called deterministic because if you repeat the fiber tracking
# (keeping all the inputs the same) you will get exactly the same set of
# streamlines. We can save the streamlines as a Tractogram file so it can be
# loaded into other software for visualization or further analysis.

sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_tractogram(sft, "tractogram_EuDX.trx")

###############################################################################
# Handling Fiber Crossings with ``max_cross``
# -------------------------------------------
# A key feature of EuDX is its ability to handle fiber crossings. In voxels
# where multiple fiber populations cross, FORCE extracts multiple peak
# directions. By default, EuDX tracks along all valid peak directions at each
# seed point, generating multiple streamlines per seed.
#
# The ``max_cross`` parameter controls how many crossing directions to follow
# per seed. Setting ``max_cross=1`` restricts tracking to only the primary
# (strongest) peak direction, while ``max_cross=None`` (the default) allows
# tracking along all detected peaks.

# Track only the primary peak direction at each seed
streamline_generator = eudx_tracking(
    seeds,
    stopping_criterion,
    affine,
    pam=force_pam,
    step_size=0.5,
    max_cross=1,
    random_seed=1,
)
streamlines_single = Streamlines(streamline_generator)

print(f"Streamlines with all crossings: {len(streamlines)}")
print(f"Streamlines with max_cross=1: {len(streamlines_single)}")

###############################################################################
# By comparing the two results, you can see that allowing all crossings
# produces more streamlines, which may better capture the complex white matter
# architecture but also increases the chance of false positives.
#
#
# Running EuDX via the Command Line Interface (CLI)
# -------------------------------------------------
# DIPY also provides a command line interface for fiber tracking. This is
# useful for scripting or for users who prefer working in the terminal.
# The CLI uses the ``dipy_track`` command. To run EuDX tracking from the
# command line, you need a saved PAM file (which can be generated from any
# reconstruction model), a stopping criterion image (e.g. FA map), and a
# seed mask. Then run:
#
# .. code-block:: bash
#
#     dipy_track pam.pam5 fa.nii.gz seed_mask.nii.gz \
#         --tracking_method eudx \
#         --step_size 0.5 \
#         --stopping_thr 0.25 \
#         --out_tractogram tractogram_EuDX.trx
#
# To see all available options:
#
# .. code-block:: bash
#
#     dipy_track --help
#
#
# References
# ----------
#
# .. footbibliography::
#
