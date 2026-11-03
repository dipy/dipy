"""
=========================================================
Connectivity Matrices, ROI Intersections and Density Maps
=========================================================

This example is meant to be an introduction to some of the streamline tools
available in DIPY_. Some of the functions covered in this example are
``target``, ``connectivity_matrix`` and ``density_map``. ``target`` allows one
to filter streamlines that either pass through or do not pass through some
region of the brain, ``connectivity_matrix`` groups and counts streamlines
based on where in the brain they begin and end, and finally, density map counts
the number of streamlines that pass through every voxel of some image.

To get started we'll need to have a set of streamlines to work with. We'll use
EuDX along with the CsaOdfModel to make some streamlines. Let's import the
modules and download the data we'll be using.

Let's load the necessary modules:
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.direction import peaks
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst import shm
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import eudx_tracking
from dipy.viz import actor, colormap as cmap, window

###############################################################################
# We'll be using the Stanford HARDI dataset which consists of a single
# subject's diffusion, b-values and b-vectors, T1 image and some labels in the
# same space as the T1. We'll use the ``get_fnames`` function to download the
# files we need and set the file names to variables.

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")
t1_fname = get_fnames(name="stanford_t1")

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels, labels_affine = load_nifti(label_fname)
t1_data, t1_affine = load_nifti(t1_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

###############################################################################
# .. note::
#    Affine, labels_affine and t1_affine are the same in this example, so we can
#    use any of them for the affine transformation [#]_. If it was not the
#    case, we would need to transform the streamlines to the space of the
#    labels or the T1 image.
#
#
# We've loaded an image called ``labels_img`` which is a map of tissue types
# such that every integer value in the array ``labels`` represents an
# anatomical structure or tissue type [#]_. For this example, the image was
# created so that white matter voxels have values of either 1 or 2. We'll use
# ``peaks_from_model`` to apply the ``CsaOdfModel`` to each white matter voxel
# and estimate fiber orientations which we can use for tracking. We will also
# dilate this mask by 1 voxel to ensure streamlines reach the grey matter.

white_matter = binary_dilation((labels == 1) | (labels == 2))
csamodel = shm.CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(
    model=csamodel,
    data=data,
    sphere=peaks.default_sphere,
    relative_peak_threshold=0.8,
    min_separation_angle=45,
    mask=white_matter,
)

###############################################################################
# Now we can use EuDX to track all of the white matter. To keep things
# reasonably fast we use ``density=1`` which will result in 1 seeds per
# voxel. The stopping criterion, determining when the tracking stops, is set to
# stop when the tracking exits the white matter.

seeds = utils.seeds_from_mask(white_matter, affine, density=1)
stopping_criterion = BinaryStoppingCriterion(white_matter)

streamlines_generator = eudx_tracking(
    seeds,
    stopping_criterion,
    affine,
    pam=csapeaks,
    random_seed=1,
    sphere=peaks.default_sphere,
    step_size=0.5,
)
streamlines = Streamlines(streamlines_generator)

###############################################################################
# The first of the tracking utilities we'll cover here is ``target``. This
# function takes a set of streamlines and a region of interest (ROI) and
# returns only those streamlines that pass through the ROI. The ROI should be
# an array such that the voxels that belong to the ROI are ``True`` and all
# other voxels are ``False`` (this type of binary array is sometimes called a
# mask). This function can also exclude all the streamlines that pass through
# an ROI by setting the ``include`` flag to ``False``. In this example we'll
# target the streamlines of the corpus callosum. Our ``labels`` array has a
# sagittal slice of the corpus callosum identified by the label value 2. We'll
# create an ROI mask from that label and create two sets of streamlines,
# those that intersect with the ROI and those that don't.

cc_slice = labels == 2
cc_streamlines = utils.target(streamlines, affine, cc_slice)
cc_streamlines = Streamlines(cc_streamlines)

other_streamlines = utils.target(streamlines, affine, cc_slice, include=False)
other_streamlines = Streamlines(other_streamlines)
assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)

###############################################################################
# We can use some of DIPY_'s visualization tools to display the ROI we targeted
# above and all the streamlines that pass through that ROI. The ROI is the
# yellow region near the center of the axial image.


def corpus_callosum_scene(visibility):
    """Build a scene with the T1 slices selected by ``visibility`` (x, y, z)."""
    vol_actor = actor.volume_slicer(
        t1_data, affine=affine, visibility=visibility, depth_write=True
    )
    actor.show_slices(vol_actor, affine[:3, :3] @ np.array([40, 0, 35]) + affine[:3, 3])
    cc_streamlines_actor = actor.streamlines(
        cc_streamlines,
        colors=cmap.line_colors(cc_streamlines),
        thickness=0.05,
        outline_thickness=0,
    )
    cc_ROI_actor = actor.contour_from_roi(
        cc_slice, color=(1.0, 1.0, 0.0), opacity=0.5, affine=affine
    )
    scene = window.Scene()
    scene.add(vol_actor, cc_streamlines_actor, cc_ROI_actor)
    return scene


###############################################################################
# We show both views in the same window, one scene per screen. The axial screen
# only displays the axial slice and the sagittal screen only the sagittal one.
# FURY's default camera already looks from above, so the axial gizmo is added
# without snapping the camera. For the sagittal screen we simulate a click on
# the ``L`` label of its gizmo to look at the scene from the left.

axial_scene = corpus_callosum_scene(visibility=(False, False, True))
sagittal_scene = corpus_callosum_scene(visibility=(True, False, False))

show_manager = window.ShowManager(
    title="corpuscallosum_axial_sagittal",
    size=(1600, 800),
    scene=[axial_scene, sagittal_scene],
    screen_config=[1, 1],
)
actor.gizmo(show_manager, screen=0)
sagittal_gizmo = actor.gizmo(show_manager, screen=1)
actor.click_gizmo(show_manager, sagittal_gizmo, "L")
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum Axial and Corpus Callosum Sagittal
#
#
#
# Once we've targeted the corpus callosum ROI, we might want to find out which
# regions of the brain are connected by these streamlines. To do this we can
# use the ``connectivity_matrix`` function. This function takes a set of
# streamlines and an array of labels as arguments. It returns the number of
# streamlines that start and end at each pair of labels and it can return the
# streamlines grouped by their endpoints. Notice that this function only
# considers the endpoints of each streamline.

M, grouping = utils.connectivity_matrix(
    cc_streamlines,
    affine,
    labels.astype(np.uint8),
    return_mapping=True,
    mapping_as_streamlines=True,
)
M[:3, :] = 0
M[:, :3] = 0

###############################################################################
# We've set ``return_mapping`` and ``mapping_as_streamlines`` to ``True`` so
# that ``connectivity_matrix`` returns all the streamlines in
# ``cc_streamlines`` grouped by their endpoint.
#
# Because we're typically only interested in connections between gray matter
# regions, and because the label 0 represents background and the labels 1
# and 2 represent white matter, we discard the first three rows and columns
# of the connectivity matrix.
#
# We can now display this matrix using matplotlib. We display it using a log
# scale to make small values in the matrix easier to see.

fig, ax = plt.subplots()
ax.imshow(np.log1p(M), interpolation="nearest")
fig.savefig("connectivity.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Connectivity of Corpus Callosum
#
#
#
# In our example track there are more streamlines connecting regions 11 and
# 54 than any other pair of regions. These labels represent the left and right
# superior frontal gyrus respectively. These two regions are large, close
# together, have lots of corpus callosum fibers and are easy to track so this
# result should not be a surprise to anyone.
#
# However, the interpretation of streamline counts can be tricky. The
# relationship between the underlying biology and the streamline counts will
# depend on several factors, including how the tracking was done, and the
# correct way to interpret these kinds of connectivity matrices is still an
# open question in the diffusion imaging literature.
#
# The next function we'll demonstrate is ``density_map``. This function allows
# one to represent the spatial distribution of a track by counting the density
# of streamlines in each voxel. For example, let's take the track connecting
# the left and right superior frontal gyrus.

lr_superiorfrontal_track = grouping[11, 54]
shape = labels.shape
dm = utils.density_map(lr_superiorfrontal_track, affine, shape)

###############################################################################
# Let's save this density map and the streamlines so that they can be
# visualized together. In order to save the streamlines in a ".trk" file we'll
# need to move them to "trackvis space", or the representation of streamlines
# specified by the trackvis Track File format.

# Save density map
save_nifti("lr-superiorfrontal-dm.nii.gz", dm.astype("int16"), affine)

lr_sf_trk = Streamlines(lr_superiorfrontal_track)

# Save streamlines
sft = StatefulTractogram(lr_sf_trk, hardi_img, Space.RASMM)
save_tractogram(sft, "lr-superiorfrontal.trx")

###############################################################################
# .. rubric:: Footnotes
#
# .. [#] The image `aparc-reduced.nii.gz`, which we load as ``labels_img``, is
#        a modified version of label map `aparc+aseg.mgz` created by
#        FreeSurfer_. The corpus  callosum region is a combination of the
#        FreeSurfer labels 251-255. The remaining FreeSurfer labels were
#        re-mapped and reduced so that they lie between 0 and 88. To see the
#        FreeSurfer region, label and name, represented by each value, see
#        `label_info.txt` in `~/.dipy/stanford_hardi`.
# .. [#] An affine transformation is a mapping between two coordinate systems
#        that can represent scaling, rotation, shear, translation and
#        reflection. Affine transformations are often represented using a 4x4
#        matrix where the last row of the matrix is ``[0, 0, 0, 1]``.
