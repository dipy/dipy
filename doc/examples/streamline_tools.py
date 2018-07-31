"""
.. _streamline_tools:

=========================================================
Connectivity Matrices, ROI Intersections and Density Maps
=========================================================

This example is meant to be an introduction to some of the streamline tools
available in DIPY_. Some of the functions covered in this example are
``target``, ``connectivity_matrix`` and ``density_map``. ``target`` allows one
to filter streamlines that either pass through or do not pass through some
region of the brain, ``connectivity_matrix`` groups and counts streamlines
based on where in the brain they begin and end, and finally, density map counts
the number of streamlines that pass though every voxel of some image.

To get started we'll need to have a set of streamlines to work with. We'll use
EuDX along with the CsaOdfModel to make some streamlines. Let's import the
modules and download the data we'll be using.
"""

from dipy.tracking.eudx import EuDX
from dipy.reconst import peaks, shm
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines

from dipy.data import read_stanford_labels, fetch_stanford_t1, read_stanford_t1

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()

fetch_stanford_t1()
t1 = read_stanford_t1()
t1_data = t1.get_data()

"""
We've loaded an image called ``labels_img`` which is a map of tissue types such
that every integer value in the array ``labels`` represents an anatomical
structure or tissue type [#]_. For this example, the image was created so that
white matter voxels have values of either 1 or 2. We'll use
``peaks_from_model`` to apply the ``CsaOdfModel`` to each white matter voxel
and estimate fiber orientations which we can use for tracking.
"""

white_matter = (labels == 1) | (labels == 2)
csamodel = shm.CsaOdfModel(gtab, 6)
csapeaks = peaks.peaks_from_model(model=csamodel,
                                  data=data,
                                  sphere=peaks.default_sphere,
                                  relative_peak_threshold=.8,
                                  min_separation_angle=45,
                                  mask=white_matter)

"""
Now we can use EuDX to track all of the white matter. To keep things reasonably
fast we use ``density=2`` which will result in 8 seeds per voxel. We'll set
``a_low`` (the parameter which determines the threshold of FA/QA under which
tracking stops) to be very low because we've already applied a white matter
mask.
"""

seeds = utils.seeds_from_mask(white_matter, density=2)
streamline_generator = EuDX(csapeaks.peak_values, csapeaks.peak_indices,
                            odf_vertices=peaks.default_sphere.vertices,
                            a_low=.05, step_sz=.5, seeds=seeds)
affine = streamline_generator.affine

streamlines = Streamlines(streamline_generator, buffer_size=512)

"""
The first of the tracking utilities we'll cover here is ``target``. This
function takes a set of streamlines and a region of interest (ROI) and returns
only those streamlines that pass though the ROI. The ROI should be an array
such that the voxels that belong to the ROI are ``True`` and all other voxels
are ``False`` (this type of binary array is sometimes called a mask). This
function can also exclude all the streamlines that pass though an ROI by
setting the ``include`` flag to ``False``. In this example we'll target the
streamlines of the corpus callosum. Our ``labels`` array has a sagittal slice
of the corpus callosum identified by the label value 2. We'll create an ROI
mask from that label and create two sets of streamlines, those that intersect
with the ROI and those that don't.
"""

cc_slice = labels == 2
cc_streamlines = utils.target(streamlines, cc_slice, affine=affine)
cc_streamlines = Streamlines(cc_streamlines)

other_streamlines = utils.target(streamlines, cc_slice, affine=affine,
                                 include=False)
other_streamlines = Streamlines(other_streamlines)
assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)

"""
We can use some of DIPY_'s visualization tools to display the ROI we targeted
above and all the streamlines that pass though that ROI. The ROI is the yellow
region near the center of the axial image.
"""

from dipy.viz import window, actor
from dipy.viz.colormap import line_colors

# Enables/disables interactive visualization
interactive = False

# Make display objects
color = line_colors(cc_streamlines)
cc_streamlines_actor = actor.line(cc_streamlines, line_colors(cc_streamlines))
cc_ROI_actor = actor.contour_from_roi(cc_slice, color=(1., 1., 0.),
                                      opacity=0.5)

vol_actor = actor.slicer(t1_data)

vol_actor.display(x=40)
vol_actor2 = vol_actor.copy()
vol_actor2.display(z=35)

# Add display objects to canvas
r = window.Renderer()
r.add(vol_actor)
r.add(vol_actor2)
r.add(cc_streamlines_actor)
r.add(cc_ROI_actor)

# Save figures
window.record(r, n_frames=1, out_path='corpuscallosum_axial.png',
              size=(800, 800))
if interactive:
    window.show(r)
r.set_camera(position=[-1, 0, 0], focal_point=[0, 0, 0], view_up=[0, 0, 1])
window.record(r, n_frames=1, out_path='corpuscallosum_sagittal.png',
              size=(800, 800))
if interactive:
    window.show(r)

"""
.. figure:: corpuscallosum_axial.png
   :align: center

   **Corpus Callosum Axial**

.. include:: ../links_names.inc

.. figure:: corpuscallosum_sagittal.png
   :align: center

   **Corpus Callosum Sagittal**
"""
"""
Once we've targeted on the corpus callosum ROI, we might want to find out which
regions of the brain are connected by these streamlines. To do this we can use
the ``connectivity_matrix`` function. This function takes a set of streamlines
and an array of labels as arguments. It returns the number of streamlines that
start and end at each pair of labels and it can return the streamlines grouped
by their endpoints. Notice that this function only considers the endpoints of
each streamline.
"""

M, grouping = utils.connectivity_matrix(cc_streamlines, labels, affine=affine,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
M[:3, :] = 0
M[:, :3] = 0

"""
We've set ``return_mapping`` and ``mapping_as_streamlines`` to ``True`` so that
``connectivity_matrix`` returns all the streamlines in ``cc_streamlines``
grouped by their endpoint.

Because we're typically only interested in connections between gray matter
regions, and because the label 0 represents background and the labels 1 and 2
represent white matter, we discard the first three rows and columns of the
connectivity matrix.

We can now display this matrix using matplotlib, we display it using a log
scale to make small values in the matrix easier to see.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.imshow(np.log1p(M), interpolation='nearest')
plt.savefig("connectivity.png")

"""
.. figure:: connectivity.png
   :align: center

   **Connectivity of Corpus Callosum**

.. include:: ../links_names.inc

"""
"""
In our example track there are more streamlines connecting regions 11 and
54 than any other pair of regions. These labels represent the left and right
superior frontal gyrus respectively. These two regions are large, close
together, have lots of corpus callosum fibers and are easy to track so this
result should not be a surprise to anyone.

However, the interpretation of streamline counts can be tricky. The
relationship between the underlying biology and the streamline counts will
depend on several factors, including how the tracking was done, and the correct
way to interpret these kinds of connectivity matrices is still an open question
in the diffusion imaging literature.

The next function we'll demonstrate is ``density_map``. This function allows
one to represent the spatial distribution of a track by counting the density of
streamlines in each voxel. For example, let's take the track connecting the
left and right superior frontal gyrus.
"""

lr_superiorfrontal_track = grouping[11, 54]
shape = labels.shape
dm = utils.density_map(lr_superiorfrontal_track, shape, affine=affine)

"""
Let's save this density map and the streamlines so that they can be
visualized together. In order to save the streamlines in a ".trk" file we'll
need to move them to "trackvis space", or the representation of streamlines
specified by the trackvis Track File format.

To do that, we will use tools available in `nibabel <http://nipy.org/nibabel>`_)
"""

import nibabel as nib
from dipy.io.streamline import save_trk

# Save density map
dm_img = nib.Nifti1Image(dm.astype("int16"), hardi_img.affine)
dm_img.to_filename("lr-superiorfrontal-dm.nii.gz")

# Move streamlines to "trackvis space"
voxel_size = labels_img.header.get_zooms()
trackvis_point_space = utils.affine_for_trackvis(voxel_size)
# lr_sf_trk = utils.move_streamlines(lr_superiorfrontal_track,
#                                   trackvis_point_space, input_space=affine)

lr_sf_trk = Streamlines(lr_superiorfrontal_track)

# Save streamlines
save_trk("lr-superiorfrontal.trk", lr_sf_trk, shape=shape, vox_size=voxel_size, affine=affine)

"""
Let's take a moment here to consider the representation of streamlines used in
DIPY. Streamlines are a path though the 3D space of an image represented by a
set of points. For these points to have a meaningful interpretation, these
points must be given in a known coordinate system. The ``affine`` attribute of
the ``streamline_generator`` object specifies the coordinate system of the
points with respect to the voxel indices of the input data.
``trackvis_point_space`` specifies the trackvis coordinate system with respect
to the same indices. The ``move_streamlines`` function returns a new set of
streamlines from an existing set of streamlines in the target space. The
target space and the input space must be specified as affine transformations
with respect to the same reference [#]_. If no input space is given, the input
space will be the same as the current representation of the streamlines, in
other words the input space is assumed to be ``np.eye(4)``, the 4-by-4 identity
matrix.

All of the functions above that allow streamlines to interact with volumes take
an affine argument. This argument allows these functions to work with
streamlines regardless of their coordinate system. For example even though we
moved our streamlines to "trackvis space", we can still compute the density map
as long as we specify the right coordinate system.
"""

dm_trackvis = utils.density_map(lr_sf_trk, shape, affine=np.eye(4))
assert np.all(dm == dm_trackvis)

"""
This means that streamlines can interact with any image volume, for example a
high resolution structural image, as long as one can register that image to
the diffusion images and calculate the coordinate system with respect to that
image.
"""
"""
.. rubric:: Footnotes

.. [#] The image `aparc-reduced.nii.gz`, which we load as ``labels_img``, is a
       modified version of label map `aparc+aseg.mgz` created by `FreeSurfer
       <https://surfer.nmr.mgh.harvard.edu/>`_. The corpus callosum region is a
       combination of the FreeSurfer labels 251-255. The remaining FreeSurfer
       labels were re-mapped and reduced so that they lie between 0 and 88. To
       see the FreeSurfer region, label and name, represented by each value see
       `label_info.txt` in `~/.dipy/stanford_hardi`.
.. [#] An affine transformation is a mapping between two coordinate systems
       that can represent scaling, rotation, sheer, translation and reflection.
       Affine transformations are often represented using a 4x4 matrix where the
       last row of the matrix is ``[0, 0, 0, 1]``.
"""
