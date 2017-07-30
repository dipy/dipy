"""
.. _sfm-track:

==================================================
Tracking with the Sparse Fascicle Model
==================================================

Tracking requires a per-voxel model. Here, the model is the Sparse Fascicle
Model (SFM), described in [Rokem2015]_. This model reconstructs the diffusion
signal as a combination of the signals from different fascicles (see also
:ref:`sfm-reconst`).

To begin, we read the Stanford HARDI data set into memory:
"""

from dipy.data import read_stanford_labels
hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine

"""
This data set provides a label map (generated using `FreeSurfer
<https://surfer.nmr.mgh.harvard.edu/>`_), in which the white matter voxels are
labeled as either 1 or 2:
"""

white_matter = (labels == 1) | (labels == 2)

"""
The first step in tracking is generating a model from which tracking directions
can be extracted in every voxel.

For the SFM, this requires first that we define a canonical response function
that will be used to deconvolve the signal in every voxel
"""

from dipy.reconst.csdeconv import auto_response
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)


"""
We initialize an SFM model object, using this response function and using the
default sphere (362  vertices, symmetrically distributed on the surface of the
sphere):
"""

from dipy.data import get_sphere
sphere = get_sphere()
from dipy.reconst import sfm
sf_model = sfm.SparseFascicleModel(gtab, sphere=sphere,
                                   l1_ratio=0.5, alpha=0.001,
                                   response=response[0])

"""
We fit this model to the data in each voxel in the white-matter mask, so that
we can use these directions in tracking:
"""

from dipy.direction.peaks import peaks_from_model

pnm = peaks_from_model(sf_model, data, sphere,
                       relative_peak_threshold=.5,
                       min_separation_angle=25,
                       mask=white_matter,
                       parallel=True
                       )

"""
A ThresholdTissueClassifier object is used to segment the data to track only
through areas in which the Generalized Fractional Anisotropy (GFA) is
sufficiently high.
"""

from dipy.tracking.local import ThresholdTissueClassifier
classifier = ThresholdTissueClassifier(pnm.gfa, .25)

"""
Tracking will be started from a set of seeds evenly distributed in the white
matter:
"""

from dipy.tracking import utils
seeds = utils.seeds_from_mask(white_matter, density=[2, 2, 2], affine=affine)

"""
For the sake of brevity, we will take only the first 1000 seeds, generating
only 1000 streamlines. Remove this line to track from many more points in all of
the white matter
"""

seeds = seeds[:1000]

"""
We now have the necessary components to construct a tracking pipeline and
execute the tracking
"""

from dipy.tracking.local import LocalTracking
streamlines = LocalTracking(pnm, classifier, seeds, affine, step_size=.5)

streamlines = list(streamlines)

"""
Next, we will create a visualization of these streamlines, relative to this
subject's T1-weighted anatomy:
"""

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from dipy.data import read_stanford_t1
from dipy.tracking.utils import move_streamlines
from numpy.linalg import inv
t1 = read_stanford_t1()
t1_data = t1.get_data()
t1_aff = t1.affine
color = line_colors(streamlines)

"""
To speed up visualization, we will select a random sub-set of streamlines to
display. This is particularly important, if you track from seeds throughout the
entire white matter, generating many streamlines. In this case, for
demonstration purposes, we subselect 900 streamlines.
"""

from dipy.tracking.streamline import select_random_set_of_streamlines
plot_streamlines = select_random_set_of_streamlines(streamlines, 900)

streamlines_actor = fvtk.streamtube(
    list(move_streamlines(plot_streamlines, inv(t1_aff))),
    line_colors(streamlines), linewidth=0.1)

vol_actor = fvtk.slicer(t1_data)

vol_actor.display(40, None, None)
vol_actor2 = vol_actor.copy()
vol_actor2.display(None, None, 35)

ren = fvtk.ren()
fvtk.add(ren, streamlines_actor)
fvtk.add(ren, vol_actor)
fvtk.add(ren, vol_actor2)

fvtk.record(ren, n_frames=1, out_path='sfm_streamlines.png',
            size=(800, 800))

"""
.. figure:: sfm_streamlines.png
   :align: center

   **Sparse Fascicle Model tracks**

Finally, we can save these streamlines to a 'trk' file, for use in other
software, or for further analysis.
"""

from dipy.io.trackvis import save_trk
save_trk("sfm_detr.trk", streamlines, affine, labels.shape)

"""
References
----------

.. [Rokem2015] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
   N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell (2015). Evaluating
   the accuracy of diffusion MRI models in white matter. PLoS ONE 10(4):
   e0123272. doi:10.1371/journal.pone.0123272

"""
