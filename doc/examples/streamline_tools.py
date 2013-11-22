"""

==================================================
Using Dipy to Target, Group, and Count Streamlines
==================================================

This example is meant to be an introduction to some of the streamline tools
available in dipy. Some of the functions covered in this example are
``target``, ``connectivity_matrix`` and ``density map``. ``target`` allows one
to filter streamlines that either pass though or do not pass through some
region of the brain, ``connectivity__matrix`` groups and counts streamlines
based on where in the brain be begin and end, and finally, density map counts
the number of streamlines that pass though every voxel of some image.

To get started you'll need to have a set of streamlines to work with. We'll use
EuDx along with the CsaOdfModel to make some streamlines. Lets import the
modules and download the data we'll be using.
"""

from dipy.tracking.eudx import EuDX
from dipy.reconst import peaks, shm
from dipy.tracking import utils

from dipy.data import read_stanford_labels

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()

"""
We've loaded an image called ``labels_img`` which is a map of tissue types such
that every integer value in the array ``labels`` represents a anatomical
structure or tissue type [#]_. For this example, the image was created so that
white matter voxels have values of either 1 or 2. We'll use ``peak_from_model``
to apply the ``CsaOdfModel`` to each white matter voxel and estimate fiber
orientations which we can use for tracking.
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
Now we can use EuDx track all of the white matter. To keep things
reasonably fast we use 1 seed per voxel here.
"""

seeds = utils.seeds_from_mask(white_matter, density=1)
streamlines = EuDX(csapeaks.peak_values,
                   csapeaks.peak_indices,
                   odf_vertices=peaks.default_sphere.vertices,
                   a_low=0.,
                   step_sz=.5,
                   seeds=seeds)

"""
The first of the tracking utilities we'll cover here is ``target``. This
function takes a set of streamlines and an rio and returns only those that pass
though the roi. The roi should be an array such that the voxels that belong to
the roi are True and all other voxels are False. In this example we'll target
the streamlines of the corpus callosum. Our ``labels`` array has a sagital
slice of the corpus callosum identified by the label value 2. We'll create an
roi mask from that label and target that roi.
"""

cc_slice = labels == 2
cc_streamlines = utils.target(streamlines, cc_slice)
cc_streamlines = list(cc_streamlines)

"""
Lets take a quick look at what these streamlines look like. We can display the
streamlines using dipy's viz module in ``dipy.viz``. As you can see, from a
whole brain set of streamlines, we've selected only those streamlines that pass
thought the corpus callosum, shown in yellow.
"""

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

# Make display objects
color = line_colors(cc_streamlines)
cc_streamlines_actor = fvtk.line(cc_streamlines, line_colors(cc_streamlines))
cc_roi_actor = fvtk.contour(cc_slice, levels=[1], colors=[(1., 1., 0.)],
                            opacities=[1.])

# Add display objects to canvas
r = fvtk.ren()
fvtk.add(r, cc_streamlines_actor)
fvtk.add(r, cc_roi_actor)

# Save figures
fvtk.record(r, n_frames=1, out_path='corpuscollosum_axial.png',
            size=(800, 800))
fvtk.camera(r, [-1, 0, 0], [0, 0, 0], viewup=[0, 0, 1])
fvtk.record(r, n_frames=1, out_path='corpuscollosum_sagital.png',
            size=(800, 800))

"""
.. figure:: corpuscollosum_axial.png
   :align: center

   **Corpus Collosum Axial**

.. include:: ../links_names.inc

.. figure:: corpuscollosum_sagital.png
   :align: center

   **Corpus Collosum Sagital**
"""

"""
Once we've targeted on the corpus collosum roi, we might want to find out
which regions of the brain are connected by these streamlines. To do this we
can use the connectivity_matrix function. This function takes a set of
streamlines and an array of labels as arguments. It returns the number of
streamlines that start and end at each pair of labels and it can return the
streamlines grouped by their endpoints. Notice that this function only
considers the endpoints of each streamline.
"""

M, grouping = utils.connectivity_matrix(cc_streamlines, labels,
                                        symmetric=True,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
M[:3, :] = 0
M[:, :3] = 0

"""
We've set ``return_mapping`` and ``mapping_as_streamlines`` to True so that
connectivity_matrix returns all the streamlines in cc_streamlines grouped by
their endpoint.

We've set ``symmetric`` to True so that the start and end points are treated
the same and our connectivity matrix will be symmetric.

Because we're typically only interested int gray-gray connections, and because
the label 0 represents background and the labels 1 and 2 represent white
matter, we then discard the first three rows and columns of the connectivity
matrix.

We can now display this matrix using matplotlib, we dispaly it using a log
scale to make small values in the matrix easier to see.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.imshow(np.log1p(M), interpolation='nearest')
plt.title("Connectivity of Corpus Collosum Streamlines")
plt.savefig("connectivity.png")

"""
.. figure:: connectivity.png
   :align: center

   **Connectivity of Corpus Collosum**

.. include:: ../links_names.inc

"""

"""
In our example track there are more streamlines connecting regions 11 and
54 than any other pair of regions. These labels represent the left and right
superior frontal gyrus respectively. These two regions are large, close
together, have lots of corpus collosum fibers and easy to track so this result
should not be a surprise to anyone.

Lets lake a look at what the streamlines that connect these two regions look
like.
"""


"""
In order to demonstrate the density_map function, we'll use the streamlines
connecting the left and right superior frontal gyrus. We can get these
streamlines from the dictionary ``grouping``. We'll then give those streamlines
as an argument to the ``density_map`` function which will count the number of
streamlines that pass though each voxel. We'll also need give ``density_map``
the dimensions of the image.
"""

lr_superiorfrontal_track = grouping[11, 54]
shape = labels.shape
dm = utils.density_map(lr_superiorfrontal_track, shape)

"""
Lets save this density map and the streamlines so that they can be
visualized together. In order to save the streamlines in a ".trk" file we'll
need to move them to "trackvis space".
"""

import nibabel as nib

voxel_size = labels_img.get_header().get_zooms()
trackvis_header = nib.trackvis.empty_header()
trackvis_header['voxel_size'] = voxel_size
trackvis_header['dim'] = shape
trackvis_header['voxel_order'] = "RAS"

lr_sf_trk = [(sl + .5) * voxel_size for sl in lr_superiorfrontal_track]
for_save = [(sl, None, None) for sl in lr_sf_trk]
nib.trackvis.write("lr-superiorfrontal.trk", for_save, trackvis_header)

dm_img = nib.Nifti1Image(dm.astype("int16"), hardi_img.get_affine())
dm_img.to_filename("lr-superiorfrontal-dm.nii.gz")

"""
Since we have the streamlines in "trackvis space" lets take a moment to
consider the representation of streamlines used in dipy. Streamlines are simply
a sequence of points in 3d space. These points can be represented using
different coordinate systems. So far in this example, all points have been in
the "voxel space" of the data that was used to create the streamlines. That is,
the point [0., 0., 0.] is at the center of the voxel [0, 0, 0]. And the point
[0., 0., .5] is half way between voxels [0, 0, 0] and [0, 0, 1].

All of the streamline tools that allow streamlines to interact with images must
be able to map between the points of the streamlines and the indices of the
image arrays. In order to do this, they take two optional keyword arguments,
``affine`` and ``voxel_size``, which can be used to specify the coordinate
system being used. If neither of these arugments is specified, the streamlines
should be in voxel coordinates. If the ``affine`` argument is specified any
affine mapping between voxel coordinates and streamline points can be used.
The ``voxel_size`` argument is meant to be used with streamlines in "trackvis
space" along with the ``voxel_size`` filed of the trackvis header.

The streamlines in ``lr_sf_trk`` were moved to "trackvis space" for saving.
If we use them to make a density map we'll get the same result as before as
long as we don't forget to specify the coordinate system.
"""

voxel_size = trackvis_header['voxel_size']
dm_using_voxel_size = utils.density_map(lr_sf_trk, shape, voxel_size=voxel_size)
assert np.all(dm == dm_using_voxel_size)

"""
We can also use the affine to specify the coordinate system, we just have
to be careful to build the right affine. Here's how you can build an affine for
"trackvis space" coordinates.
"""

affine = np.eye(4)
affine[[0, 1, 2], [0, 1, 2]] = voxel_size
affine[:3, 3] = voxel_size / 2.

dm_using_affine = utils.density_map(lr_sf_trk, shape, affine=affine)
assert np.all(dm == dm_using_affine)

"""
.. rubric:: Footnotes

.. [#] The image `aparc-reduced.nii.gz`, which we load as ``labels_img``, was
    is a modified versioin of label map `aparc+aseg.mgz` created by freesurfer.
    The corpus callosum region is a combination of the freesurfer labels
    251-255.  The remaining freesurfer labels were re-mapped and reduced so
    that they lie between 0 and 88. To see the freesurfer region, label and
    name, represented by each value see `label_info.txt` in
    `~/.dipy/stanford_hardi`.
..
"""
