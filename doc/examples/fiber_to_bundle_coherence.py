# -*- coding: utf-8 -*-
"""
==================================
Fiber to bundle coherence measures
==================================

This demo presents the fiber to bundle coherence (FBC) quantitative
measure of the alignment of each fiber with the surrounding fiber bundles
[Meesters2016]_. These measures are useful in “cleaning” the results of
tractography algorithms, since low FBCs indicate which fibers are isolated and
poorly aligned with their neighbors, as shown in the figure below.

.. _fiber_to_bundle_coherence:

.. figure:: _static/fbc_illustration.png
   :scale: 60 %
   :align: center

   On the left this figure illustrates (in 2D) the contribution of two fiber
   points to the kernel density estimator. The kernel density estimator is the
   sum over all such locally aligned kernels. The local fiber to bundle
   coherence shown on the right color-coded for each fiber, is obtained by
   evaluating the kernel density estimator along the fibers. One spurious
   fiber is present which is isolated and badly aligned with the other fibers,
   and can be identified by a low LFBC value in the region where it deviates
   from the bundle. Figure adapted from [Portegies2015]_.

Here we implement FBC measures based on kernel density estimation in the
non-flat 5D position-orientation domain. First we compute the kernel density
estimator induced by the full lifted output (defined in the space of positions
and orientations) of the tractography. Then, the Local FBC (LFBC) is the
result of evaluating the estimator along each element of the lifted fiber.
A whole fiber measure, the relative FBC (RFBC), is calculated
by the minimum of the moving average LFBC along the fiber.
Details of the computation of FBC can be found in [Portegies2015]_.

"""

"""
The FBC measures are evaluated on the Stanford HARDI dataset
(150 orientations, b=2000 $s/mm^2$) which is one of the standard example
datasets in DIPY_.
"""

import numpy as np
from dipy.data import (read_stanford_labels, fetch_stanford_t1,
                       read_stanford_t1)

# Fix seed
np.random.seed(1)

# Read data
hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine
fetch_stanford_t1()
t1 = read_stanford_t1()
t1_data = t1.get_data()

# Select a relevant part of the data (left hemisphere)
# Coordinates given in x bounds, y bounds, z bounds
dshape = data.shape[:-1]
xa, xb, ya, yb, za, zb = [15, 42, 10, 65, 18, 65]
data_small = data[xa:xb, ya:yb, za:zb]
selectionmask = np.zeros(dshape, 'bool')
selectionmask[xa:xb, ya:yb, za:zb] = True

"""
The data is first fitted to Constant Solid Angle (CDA) ODF Model. CSA is a
good choice to estimate general fractional anisotropy (GFA), which the tissue
classifier can use to restrict fiber tracking to those areas where the ODF
shows significant restricted diffusion, thus creating a region-of-interest in
which the computations are done.
"""

# Perform CSA
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.6,
                             min_separation_angle=45,
                             mask=selectionmask)

# Tissue classifier
from dipy.tracking.local import ThresholdTissueClassifier

classifier = ThresholdTissueClassifier(csa_peaks.gfa, 0.25)

"""
In order to perform probabilistic fiber tracking we first fit the data to the
Constrained Spherical Deconvolution (CSD) model in DIPY. This model represents
each voxel in the data set as a collection of small white matter fibers with
different orientations. The density of fibers along each orientation is known
as the Fiber Orientation Distribution (FOD), used in the fiber tracking.
"""

# Perform CSD on the original data
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit = csd_model.fit(data_small)
csd_fit_shm = np.lib.pad(csd_fit.shm_coeff, ((xa, dshape[0]-xb),
                                             (ya, dshape[1]-yb),
                                             (za, dshape[2]-zb),
                                             (0, 0)), 'constant')

# Probabilistic direction getting for fiber tracking
from dipy.direction import ProbabilisticDirectionGetter

prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit_shm,
                                                    max_angle=30.,
                                                    sphere=default_sphere)

"""
The optic radiation is reconstructed by tracking fibers from the calcarine
sulcus (visual cortex V1) to the lateral geniculate nucleus (LGN). We seed
from the calcarine sulcus by selecting a region-of-interest (ROI) cube of
dimensions 3x3x3 voxels.
"""

# Set a seed region region for tractography.
from dipy.tracking import utils

mask = np.zeros(data.shape[:-1], 'bool')
rad = 3
mask[26-rad:26+rad, 29-rad:29+rad, 31-rad:31+rad] = True
seeds = utils.seeds_from_mask(mask, density=[4, 4, 4], affine=affine)

"""
Local Tracking is used for probabilistic tractography which takes the
direction getter along with the classifier and seeds as input.
"""

# Perform tracking using Local Tracking
from dipy.tracking.local import LocalTracking

streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)

# Compute streamlines.
from dipy.tracking.streamline import Streamlines
streamlines = Streamlines(streamlines_generator)

"""
In order to select only the fibers that enter into the LGN, another ROI is
created from a cube of size 5x5x5 voxels. The near_roi command is used to find
the fibers that traverse through this ROI.
"""

# Set a mask for the lateral geniculate nucleus (LGN)
mask_lgn = np.zeros(data.shape[:-1], 'bool')
rad = 5
mask_lgn[35-rad:35+rad, 42-rad:42+rad, 28-rad:28+rad] = True

# Select all the fibers that enter the LGN and discard all others
filtered_fibers2 = utils.near_roi(streamlines, mask_lgn, tol=1.8,
                                  affine=affine)

sfil = []
for i in range(len(streamlines)):
    if filtered_fibers2[i]:
        sfil.append(streamlines[i])
streamlines = Streamlines(sfil)

"""
Inspired by [Rodrigues2010]_, a lookup-table is created, containing rotated
versions of the fiber propagation kernel :math:`P_t` [DuitsAndFranken2011]_
rotated over a discrete set of orientations. See the
`Contextual enhancement example <http://nipy.org/dipy/examples_built/contextual_enhancement.html>`_
for more details regarding the kernel. In order to ensure rotationally
invariant processing, the discrete orientations are required to be equally
distributed over a sphere. By default, a sphere with 100 directions is used
obtained from electrostatic repulsion in DIPY.
"""

# Compute lookup table
from dipy.denoise.enhancement_kernel import EnhancementKernel

D33 = 1.0
D44 = 0.02
t = 1
k = EnhancementKernel(D33, D44, t)

"""
The FBC measures are now computed, taking the tractography results and the
lookup tables as input.
"""

# Apply FBC measures
from dipy.tracking.fbcmeasures import FBCMeasures

fbc = FBCMeasures(streamlines, k)

"""
After calculating the FBC measures, a threshold can be chosen on the relative
FBC (RFBC) in order to remove spurious fibers. Recall that the relative FBC
(RFBC) is calculated by the minimum of the moving average LFBC along the fiber.
In this example we show the results for threshold 0 (i.e. all fibers are
included) and 0.2 (removing the 20 percent most spurious fibers).
"""

# Calculate LFBC for original fibers
fbc_sl_orig, clrs_orig, rfbc_orig = \
  fbc.get_points_rfbc_thresholded(0, emphasis=0.01)

# Apply a threshold on the RFBC to remove spurious fibers
fbc_sl_thres, clrs_thres, rfbc_thres = \
  fbc.get_points_rfbc_thresholded(0.125, emphasis=0.01)

"""
The results of FBC measures are visualized, showing the original fibers
colored by LFBC (see :ref:`optic_radiation_before_cleaning`), and the fibers
after the cleaning procedure via RFBC thresholding (see
:ref:`optic_radiation_after_cleaning`).
"""

# Visualize the results
from dipy.viz import window, actor

# Enables/disables interactive visualization
interactive = False

# Create renderer
ren = window.Renderer()

# Original lines colored by LFBC
lineactor = actor.line(fbc_sl_orig, clrs_orig, linewidth=0.2)
ren.add(lineactor)

# Horizontal (axial) slice of T1 data
vol_actor1 = actor.slicer(t1_data, affine=affine)
vol_actor1.display(z=20)
ren.add(vol_actor1)

# Vertical (sagittal) slice of T1 data
vol_actor2 = actor.slicer(t1_data, affine=affine)
vol_actor2.display(x=35)
ren.add(vol_actor2)

# Show original fibers
ren.set_camera(position=(-264, 285, 155), focal_point=(0, -14, 9), view_up=(0, 0, 1))
window.record(ren, n_frames=1, out_path='OR_before.png', size=(900, 900))
if interactive:
    window.show(ren)

# Show thresholded fibers
ren.rm(lineactor)
ren.add(actor.line(fbc_sl_thres, clrs_thres, linewidth=0.2))
window.record(ren, n_frames=1, out_path='OR_after.png', size=(900, 900))
if interactive:
    window.show(ren)

"""
.. _optic_radiation_before_cleaning:

.. figure:: OR_before.png
   :align: center

   The optic radiation obtained through probabilistic tractography colored by
   local fiber to bundle coherence.

.. _optic_radiation_after_cleaning:

.. figure:: OR_after.png
   :align: center

   The tractography result is cleaned (shown in bottom) by removing fibers
   with a relative FBC (RFBC) lower than the threshold :math:`\tau = 0.2`.

Acknowledgments
---------------
The techniques are developed in close collaboration with Pauly Ossenblok of
the Academic Center of Epileptology Kempenhaeghe & Maastricht UMC+.

References
----------

.. [Meesters2016] S. Meesters, G. Sanguinetti, E. Garyfallidis, J. Portegies,
   P. Ossenblok, R. Duits. (2016) Cleaning output of tractography via fiber to
   bundle coherence, a new open source implementation. Human Brain Mapping
   Conference 2016.

.. [Portegies2015] J. Portegies, R. Fick, G. Sanguinetti, S. Meesters,
   G.Girard, and R. Duits. (2015) Improving Fiber Alignment in HARDI by
   Combining Contextual PDE flow with Constrained Spherical Deconvolution. PLoS
   One.

.. [DuitsAndFranken2011] R. Duits and E. Franken (2011) Left-invariant
   diffusions on the space of positions and orientations and their application
   to crossing-preserving smoothing of HARDI images. International Journal of
   Computer Vision, 92:231-264.

.. [Rodrigues2010] P. Rodrigues, R. Duits, B. Romeny, A. Vilanova (2010).
   Accelerated Diffusion Operators for Enhancing DW-MRI. Eurographics Workshop
   on Visual Computing for Biology and Medicine. The Eurographics Association.

.. include:: ../links_names.inc

"""
