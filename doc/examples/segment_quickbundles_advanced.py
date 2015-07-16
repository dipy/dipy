"""
====================================================
Tractography Clustering with QuickBundles (advanced)
====================================================

The following examples show advanced usages of QuickBundles [Garyfallidis12]_
and the clustering framework. If you are not familiar with either one you should
check :ref:`example_segment_quickbundles` for an introduction to tractography
clustering with QuickBundles, or check :ref:`clustering-framework` to have a basic
understanding of how the clustering framework works in Dipy.

First import the necessary modules and load a small streamline bundle.
"""

from nibabel import trackvis as tv
from dipy.segment.clustering import QuickBundles
from dipy.data import get_data

fname = get_data('fornix')
streams, hdr = tv.read(fname)
streamlines = [i[0] for i in streams]

"""
QuickBundles using `ResampleFeature`
====================================
By default, QuickBundles algorithm internally uses a representation of
streamlines that are either downsampled or upsampled so they have 12 points.
To tell QuickBundles to use a different number of points when resampling, one
needs to use the `ResampleFeature` feature.

Perform QuickBundles clustering using the MDF metric and a 10mm distance
threshold on streamlines that will be internally resampled to 24 points.
*Note `ResampleFeature` performs the resampling on the fly so there are no
permanent modifications made to your streamlines.*
"""

from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
feature = ResampleFeature(nb_points=24)
metric = AveragePointwiseEuclideanMetric(feature=feature)
qb = QuickBundles(threshold=10., metric=metric)
clusters = qb.cluster(streamlines)


"""
.. include:: ../links_names.inc

.. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                    tractography simplification, Frontiers in Neuroscience, vol
                    6, no 175, 2012.
"""
