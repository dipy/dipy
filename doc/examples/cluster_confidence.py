"""
==================================
Calculation of Outliers with Cluster Confidence Index
==================================

This is an outlier scoring method that compares the pathways of each streamline
in a bundle (pairwise) and scores each streamline by how many other streamlines
have similar pathways. The details can be found in [Jordan_2018_plm]_.

"""

from dipy.data import read_stanford_labels
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.local import ThresholdTissueClassifier
from dipy.tracking import utils
from dipy.tracking.local import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.viz import actor, window
from dipy.tracking.utils import length

import matplotlib.pyplot as plt
import matplotlib

from dipy.tracking.streamline import cluster_confidence


"""
First, we need to generate some streamlines. For a more complete
description of these steps, please refer to the CSA Probabilistic Tracking and
the Visualization of ROI Surface Rendered with Streamlines Tutorials.
 """


hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine
white_matter = (labels == 1) | (labels == 2)
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=white_matter)
classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)


"""
We will use a slice of the anatomically-based corpus callosum ROI as our
seed mask to demonstrate the method.
 """


# Make a corpus callosum seed mask for tracking
seed_mask = labels == 2
seeds = utils.seeds_from_mask(seed_mask, density=[1, 1, 1], affine=affine)
# Make a streamline bundle model of the corpus callosum ROI connectivity
streamlines = LocalTracking(csa_peaks, classifier, seeds, affine,
                            step_size=2)
streamlines = Streamlines(streamlines)


"""
We do not want our results inflated by short streamlines, so we remove
streamlines shorter than 40mm prior to calculating the CCI.
"""

lengths = list(length(streamlines))
long_streamlines = Streamlines()
for i, sl in enumerate(streamlines):
    if lengths[i] > 40:
        long_streamlines.append(sl)


"""
Now we calculate the Cluster Confidence Index using the corpus callosum
streamline bundle and visualize them.
"""


cci = cluster_confidence(long_streamlines)

# Visualize the streamlines, colored by cci
ren = window.Renderer()

hue = [0.5, 1]
saturation = [0.0, 1.0]

lut_cmap = actor.colormap_lookup_table(scale_range=(cci.min(), cci.max()/4),
                                       hue_range=hue,
                                       saturation_range=saturation)

bar3 = actor.scalar_bar(lut_cmap)
ren.add(bar3)

stream_actor = actor.line(long_streamlines, cci, linewidth=0.1,
                          lookup_colormap=lut_cmap)
ren.add(stream_actor)


"""
If you set interactive to True (below), the rendering will pop up in an
interactive window.
"""


interactive = False
if interactive:
    window.show(ren)
window.record(ren, n_frames=1, out_path='cci_streamlines.png',
              size=(800, 800))

"""
.. figure:: cci_streamlines.png
   :align: center

   Cluster Confidence Index of corpus callosum dataset.


If you think of each streamline as a sample of a potential pathway through a
complex landscape of white matter anatomy probed via water diffusion,
intuitively we have more confidence that pathways represented by many samples
(streamlines) reflect a more stable representation of the underlying phenomenon
we are trying to model (anatomical landscape) than do lone samples.

The CCI provides a voting system where by each streamline (within a set
tolerance) gets to vote on how much support it lends to. Outlier pathways score
relatively low on CCI, since they do not have many streamlines voting for them.
These outliers can be removed by thresholding on the CCI metric.

"""


fig, ax = plt.subplots(1)
ax.hist(cci, bins=100, histtype='step')
ax.set_xlabel('CCI')
ax.set_ylabel('# streamlines')
fig.savefig('cci_histogram.png')


"""
.. figure:: cci_histogram.png
   :align: center

   Histogram of Cluster Confidence Index values.

Now we threshold the CCI, defining outliers as streamlines that score below 1.

"""

keep_streamlines = Streamlines()
for i, sl in enumerate(long_streamlines):
    if cci[i] >= 1:
        keep_streamlines.append(sl)

# Visualize the streamlines we kept
ren = window.Renderer()

keep_streamlines_actor = actor.line(keep_streamlines, linewidth=0.1)

ren.add(keep_streamlines_actor)


interactive = False
if interactive:
    window.show(ren)
window.record(ren, n_frames=1, out_path='filtered_cci_streamlines.png',
              size=(800, 800))

"""

.. figure:: filtered_cci_streamlines.png
   :align: center

   Outliers, defined as streamlines scoring CCI < 1, were excluded.


References
----------

.. [Jordan_2018_plm] Jordan, K., Amirbekian, B., Keshavan, A., Henry, R.G.
"Cluster Confidence Index: A Streamline‐Wise Pathway Reproducibility Metric
for Diffusion‐Weighted MRI Tractography", Journal of Neuroimaging, 2017.

.. include:: ../links_names.inc

"""
