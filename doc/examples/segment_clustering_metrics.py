"""
===========================================
Tractography Clustering - Available Metrics
===========================================

This page lists available metrics that can be used by the tractography
clustering framework. For every metric a brief description is provided
explaining: what it does, when it's useful and how to use it. If you are not
familiar with the tractography clustering framework, check this tutorial
:ref:`clustering-framework`.

.. contents:: Available Metrics
    :local:
    :depth: 1

**Note**:
All examples assume a function `get_streamlines` exists. We defined here a
simple function to do so. It imports the necessary modules and load a small
streamline bundle.
"""


def get_streamlines():
    from nibabel import trackvis as tv
    from dipy.data import get_data

    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    return streamlines

"""
.. _clustering-examples-AveragePointwiseEuclideanMetric:

Average of Pointwise Euclidean Metric
=====================================
**What:** Instances of `AveragePointwiseEuclideanMetric` first compute the
pointwise Euclidean distance between two sequences *of same length* then
return the average of those distances. This metric takes as inputs two features
that are sequences containing the same number of elements.

**When:** By default the `QuickBundles` clustering will resample your
streamlines on-the-fly so they have 12 points. If for some reason you want
to avoid this and you made sure all your streamlines have already the same
number of points, you can manually provide an instance of
`AveragePointwiseEuclideanMetric` to `QuickBundles`. Since the default
`Feature` is the `IdentityFeature` the streamlines won't be resampled thus
saving some computational time.

**Note:** Inputs must be sequences of same length.
"""

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

# Make sure our streamlines have the same number of points.
from dipy.tracking.streamline import set_number_of_points
streamlines = set_number_of_points(streamlines, nb_points=12)

# Create the instance of `AveragePointwiseEuclideanMetric` to use.
metric = AveragePointwiseEuclideanMetric()
qb = QuickBundles(threshold=10., metric=metric)
clusters = qb.cluster(streamlines)

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))

"""

::

    Nb. clusters: 4

    Cluster sizes: [64, 191, 44, 1]

.. _clustering-examples-SumPointwiseEuclideanMetric:

Sum of Pointwise Euclidean Metric
=================================
**What:** Instances of `SumPointwiseEuclideanMetric` first compute the
pointwise Euclidean distance between two sequences *of same length* then
return the sum of those distances.

**When:** This metric mainly exists because it is used internally by
`AveragePointwiseEuclideanMetric`.

**Note:** Inputs must be sequences of same length.
"""

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import SumPointwiseEuclideanMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

# Make sure our streamlines have the same number of points.
from dipy.tracking.streamline import set_number_of_points
nb_points = 12
streamlines = set_number_of_points(streamlines, nb_points=nb_points)

# Create the instance of `SumPointwiseEuclideanMetric` to use.
metric = SumPointwiseEuclideanMetric()
qb = QuickBundles(threshold=10.*nb_points, metric=metric)
clusters = qb.cluster(streamlines)

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))

"""

::

    Nb. clusters: 4

    Cluster sizes: [64, 191, 44, 1]


Cosine Metric
=============
**What:** Instances of `CosineMetric` compute the cosine distance between two
vectors (for more information see the
`wiki page <https://en.wikipedia.org/wiki/Cosine_similarity>`_).

**When:** This metric can be useful when you *only* need information about the
orientation of a streamline.

**Note:** Inputs must be vectors (i.e. 1D array).
"""

import numpy as np
from dipy.viz import window, actor
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import VectorOfEndpointsFeature
from dipy.segment.metric import CosineMetric

# Enables/disables interactive visualization
interactive = False

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

feature = VectorOfEndpointsFeature()
metric = CosineMetric(feature)
qb = QuickBundles(threshold=0.1, metric=metric)
clusters = qb.cluster(streamlines)

# Color each streamline according to the cluster they belong to.
colormap = actor.create_colormap(np.arange(len(clusters)))
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

# Visualization
ren = window.Renderer()
window.clear(ren)
ren.SetBackground(0, 0, 0)
ren.add(actor.streamtube(streamlines, colormap_full))
window.record(ren, out_path='cosine_metric.png', size=(600, 600))
if interactive:
    window.show(ren)

"""
.. figure:: cosine_metric.png
   :align: center

   Showing the streamlines colored according to their orientation.

.. include:: ../links_names.inc

References
----------

.. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
   tractography simplification, Frontiers in Neuroscience, vol 6, no 175, 2012.
"""
