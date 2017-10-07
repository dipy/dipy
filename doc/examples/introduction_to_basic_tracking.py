"""

.. _intro_basic_tracking:

==============================
Introduction to Basic Tracking
==============================

Local fiber tracking is an approach used to model white matter fibers by
creating streamlines from local directional information. The idea is as
follows: if the local directionality of a tract/pathway segment is known, one
can integrate along those directions to build a complete representation of that
structure. Local fiber tracking is widely used in the field of diffusion MRI
because it is simple and robust.

In order to perform local fiber tracking, three things are needed: 1) A method
for getting directions from a diffusion data set. 2) A method for identifying
different tissue types within the data set. 3) A set of seeds from which to
begin tracking.  This example shows how to combine the 3 parts described above
to create a tractography reconstruction from a diffusion data set.
"""

"""
To begin, let's load an example HARDI data set from Stanford. If you have
not already downloaded this data set, the first time you run this example you
will need to be connected to the internet and this dataset will be downloaded
to your computer.
"""

from dipy.data import read_stanford_labels

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine

"""
This dataset provides a label map in which all white matter tissues are
labeled either 1 or 2. Lets create a white matter mask to restrict tracking to
the white matter.
"""

white_matter = (labels == 1) | (labels == 2)

"""
1. The first thing we need to begin fiber tracking is a way of getting
directions from this diffusion data set. In order to do that, we can fit the
data to a Constant Solid Angle ODF Model. This model will estimate the
Orientation Distribution Function (ODF) at each voxel. The ODF is the
distribution of water diffusion as a function of direction. The peaks of an ODF
are good estimates for the orientation of tract segments at a point in the
image.
"""

from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=white_matter)

"""
2. Next we need some way of restricting the fiber tracking to areas with good
directionality information. We've already created the white matter mask,
but we can go a step further and restrict fiber tracking to those areas where
the ODF shows significant restricted diffusion by thresholding on
the general fractional anisotropy (GFA).
"""

from dipy.tracking.local import ThresholdTissueClassifier

classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

"""
3. Before we can begin tracking is to specify where to "seed" (begin) the fiber
tracking. Generally, the seeds chosen will depend on the pathways one is
interested in modeling. In this example, we'll use a $2 \times 2 \times 2$ grid
of seeds per voxel, in a sagittal slice of the corpus callosum. Tracking from
this region will give us a model of the corpus callosum tract. This slice has
label value ``2`` in the labels image.
"""

from dipy.tracking import utils

seed_mask = labels == 2
seeds = utils.seeds_from_mask(seed_mask, density=[2, 2, 2], affine=affine)

"""
Finally, we can bring it all together using ``LocalTracking``. We will then
display the resulting streamlines using the ``fvtk`` module.
"""

from dipy.tracking.local import LocalTracking
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

# Initialization of LocalTracking. The computation happens in the next step.
streamlines = LocalTracking(csa_peaks, classifier, seeds, affine, step_size=.5)

# Compute streamlines and store as a list.
streamlines = list(streamlines)

# Prepare the display objects.
color = line_colors(streamlines)

if fvtk.have_vtk:
    streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))

    # Create the 3D display.
    r = fvtk.ren()
    fvtk.add(r, streamlines_actor)

    # Save still images for this static example. Or for interactivity use
    # fvtk.show
    fvtk.record(r, n_frames=1, out_path='deterministic.png',
                size=(800, 800))

"""
.. figure:: deterministic.png
   :align: center

   **Corpus Callosum Deterministic**

We've created a deterministic set of streamlines, so called because if you
repeat the fiber tracking (keeping all the inputs the same) you will get
exactly the same set of streamlines. We can save the streamlines as a Trackvis
file so it can be loaded into other software for visualization or further
analysis.
"""

from dipy.io.trackvis import save_trk
save_trk("CSA_detr.trk", streamlines, affine, labels.shape)

"""
Next let's try some probabilistic fiber tracking. For this, we'll be using the
Constrained Spherical Deconvolution (CSD) Model. This model represents each
voxel in the data set as a collection of small white matter fibers with
different orientations. The density of fibers along each orientation is known
as the Fiber Orientation Distribution (FOD). In order to perform probabilistic
fiber tracking, we pick a fiber from the FOD at random at each new location
along the streamline. Note: one could use this model to perform deterministic
fiber tracking by always tracking along the directions that have the most
fibers.

Let's begin probabilistic fiber tracking by fitting the data to the CSD model.
"""

from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

"""
Next we'll need to make a ``ProbabilisticDirectionGetter``. Because the CSD
model represents the FOD using the spherical harmonic basis, we can use the
``from_shcoeff`` method to create the direction getter. This direction getter
will randomly sample directions from the FOD each time the tracking algorithm
needs to take another step.
"""

from dipy.direction import ProbabilisticDirectionGetter

prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)

"""
As with deterministic tracking, we'll need to use a tissue classifier to
restrict the tracking to the white matter of the brain. One might be tempted
to use the GFA of the CSD FODs to build a tissue classifier, however the GFA
values of these FODs don't classify gray matter and white matter well. We will
therefore use the GFA from the CSA model which we fit for the first section of
this example. Alternatively, one could fit a ``TensorModel`` to the data and use
the fractional anisotropy (FA) to build a tissue classifier.
"""

classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)

"""
Next we can pass this direction getter, along with the ``classifier`` and
``seeds``, to ``LocalTracking`` to get a probabilistic model of the corpus
callosum.
"""

streamlines = LocalTracking(prob_dg, classifier, seeds, affine,
                            step_size=.5, max_cross=1)

# Compute streamlines and store as a list.
streamlines = list(streamlines)

# Prepare the display objects.
color = line_colors(streamlines)

if fvtk.have_vtk:
    streamlines_actor = fvtk.line(streamlines, line_colors(streamlines))

    # Create the 3D display.
    r = fvtk.ren()
    fvtk.add(r, streamlines_actor)

    # Save still images for this static example.
    fvtk.record(r, n_frames=1, out_path='probabilistic.png',
                size=(800, 800))

"""
.. figure:: probabilistic.png
   :align: center

   Corpus callosum probabilistic tracking.
"""

save_trk("CSD_prob.trk", streamlines, affine, labels.shape)
