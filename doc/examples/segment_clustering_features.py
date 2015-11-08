"""
============================================
Tractography Clustering - Available Features
============================================

This page lists available features that can be used by the tractography
clustering framework. For every feature a brief description is provided
explaining: what it does, when it's useful and how to use it. If you are not
familiar with the tractography clustering framework, read the
:ref:`clustering-framework` first.

.. contents:: Available Features
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
.. _clustering-examples-IdentityFeature:

Identity Feature
================
**What:** Instances of `IdentityFeature` simply return the streamlines
unaltered.  In other words the features are the original data.

**When:** The QuickBundles algorithm requires streamlines to have the same
number of points. If this is the case for your streamlines, you can tell
QuickBundles to not perform resampling (see following example). The clustering
should be faster than using the default behaviour of QuickBundles since it will
require less computation (i.e. no resampling). However, it highly depends
on the number of points streamlines have. By default, QuickBundles resamples
streamlines so that they have 12 points each [Garyfallidis12]_.

*Unless stated otherwise, it is the default feature used by `Metric` objects
in the clustering framework.*
"""

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import IdentityFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

# Make sure our streamlines have the same number of points.
from dipy.tracking.streamline import set_number_of_points
streamlines = set_number_of_points(streamlines, nb_points=12)

# Create an instance of `IdentityFeature` and tell metric to use it.
feature = IdentityFeature()
metric = AveragePointwiseEuclideanMetric(feature=feature)
qb = QuickBundles(threshold=10., metric=metric)
clusters = qb.cluster(streamlines)

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))

"""

::

    Nb. clusters: 4

    Cluster sizes: [64, 191, 47, 1]


.. _clustering-examples-ResampleFeature:

Resample Feature
================
**What:** Instances of `ResampleFeature` resample streamlines to a
predetermined number of points. The resampling is done on the fly such that
there are no permanent modifications made to your streamlines.

**When:** The QuickBundles algorithm requires streamlines to have the same
number of points. By default, QuickBundles uses `ResampleFeature` to resample
streamlines so that they have 12 points each [Garyfallidis12]_. If you want to
use a different number of points for the resampling, you should provide your
own instance of `ResampleFeature` (see following example).

**Note:** Resampling streamlines has an impact on clustering results both in
term of speed and quality. Setting the number of points too low will result in
a loss of information about the shape of the streamlines. On the contrary,
setting the number of points too high will slow down the clustering process.
"""

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

# Streamlines will be resampled to 24 points on the fly.
feature = ResampleFeature(nb_points=24)
metric = AveragePointwiseEuclideanMetric(feature=feature)  # a.k.a. MDF
qb = QuickBundles(threshold=10., metric=metric)
clusters = qb.cluster(streamlines)

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))

"""

::

    Nb. clusters: 4

    Cluster sizes: [64, 191, 44, 1]

.. _clustering-examples-CenterOfMassFeature:

Center of Mass Feature
======================
**What:** Instances of `CenterOfMassFeature` compute the center of mass (also
known as center of gravity) of a set of points. This is achieved by taking the
mean of every coordinate independently (for more information see the
`wiki page <https://en.wikipedia.org/wiki/Center_of_mass>`_).

**When:** This feature can be useful when you *only* need information about the
spatial position of a streamline.

**Note:** The computed center is not guaranteed to be an existing point in the
streamline.
"""

import numpy as np
from dipy.viz import fvtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import CenterOfMassFeature
from dipy.segment.metric import EuclideanMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.


feature = CenterOfMassFeature()
metric = EuclideanMetric(feature)

qb = QuickBundles(threshold=5., metric=metric)
clusters = qb.cluster(streamlines)

# Extract feature of every streamline.
centers = np.asarray(map(feature.extract, streamlines))

# Color each center of mass according to the cluster they belong to.
rng = np.random.RandomState(42)
colormap = fvtk.create_colormap(np.arange(len(clusters)))
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

# Visualization
ren = fvtk.ren()
fvtk.clear(ren)
ren.SetBackground(0, 0, 0)
fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05))
fvtk.add(ren, fvtk.point(centers[:, 0, :], colormap_full, point_radius=0.2))
fvtk.record(ren, n_frames=1, out_path='center_of_mass_feature.png', size=(600, 600))

"""
.. figure:: center_of_mass_feature.png
   :align: center

   **Showing the center of mass of each streamline and colored according to
   the QuickBundles results**.

.. _clustering-examples-MidpointFeature:

Midpoint Feature
================
**What:** Instances of `MidpointFeature` extract the middle point of a
streamline. If there is an even number of points, the feature will then
correspond to the point halfway between the two middle points.

**When:** This feature can be useful when you *only* need information about the
spatial position of a streamline. This can also be an alternative to the
`CenterOfMassFeature` if the point extracted must be on the streamline.
"""

import numpy as np
from dipy.viz import fvtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import MidpointFeature
from dipy.segment.metric import EuclideanMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

feature = MidpointFeature()
metric = EuclideanMetric(feature)

qb = QuickBundles(threshold=5., metric=metric)
clusters = qb.cluster(streamlines)

# Extract feature of every streamline.
midpoints = np.asarray(map(feature.extract, streamlines))

# Color each midpoint according to the cluster they belong to.
rng = np.random.RandomState(42)
colormap = fvtk.create_colormap(np.arange(len(clusters)))
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

# Visualization
ren = fvtk.ren()
fvtk.clear(ren)
ren.SetBackground(0, 0, 0)
fvtk.add(ren, fvtk.point(midpoints[:, 0, :], colormap_full, point_radius=0.2))
fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05))
fvtk.record(ren, n_frames=1, out_path='midpoint_feature.png', size=(600, 600))

"""
.. figure:: midpoint_feature.png
   :align: center

   **Showing the middle point of each streamline and colored according to the
   QuickBundles results**.

.. _clustering-examples-ArcLengthFeature:

ArcLength Feature
=================
**What:** Instances of `ArcLengthFeature` compute the length of a streamline.
More specifically, this feature corresponds to the sum of the lengths of every
streamline segments.

**When:** This feature can be useful when you *only* need information about the
length of a streamline.
"""

import numpy as np
from dipy.viz import fvtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ArcLengthFeature
from dipy.segment.metric import EuclideanMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

feature = ArcLengthFeature()
metric = EuclideanMetric(feature)
qb = QuickBundles(threshold=2., metric=metric)
clusters = qb.cluster(streamlines)

# Color each streamline according to the cluster they belong to.
colormap = fvtk.create_colormap(np.ravel(clusters.centroids))
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

# Visualization
ren = fvtk.ren()
fvtk.clear(ren)
ren.SetBackground(0, 0, 0)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.record(ren, n_frames=1, out_path='arclength_feature.png', size=(600, 600))

"""
.. figure:: arclength_feature.png
   :align: center

   **Showing the streamlines colored according to their length**.

.. _clustering-examples-VectorOfEndpointsFeature:

Vector Between Endpoints Feature
================================
**What:** Instances of `VectorOfEndpointsFeature` extract the vector going
from one extremity of the streamline to the other. In other words, this feature
represents the vector beginning at the first point and ending at the last point
of the streamlines.

**When:** This feature can be useful when you *only* need information about the
orientation of a streamline.

**Note:** Since streamlines endpoints are ambiguous (e.g. the first point could
be either the beginning or the end of the streamline), one must be careful when
using this feature.
"""

import numpy as np
from dipy.viz import fvtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import VectorOfEndpointsFeature
from dipy.segment.metric import CosineMetric

# Get some streamlines.
streamlines = get_streamlines()  # Previously defined.

feature = VectorOfEndpointsFeature()
metric = CosineMetric(feature)
qb = QuickBundles(threshold=0.1, metric=metric)
clusters = qb.cluster(streamlines)

# Color each streamline according to the cluster they belong to.
colormap = fvtk.create_colormap(np.arange(len(clusters)))
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

# Visualization
ren = fvtk.ren()
fvtk.clear(ren)
ren.SetBackground(0, 0, 0)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.record(ren, n_frames=1, out_path='vector_of_endpoints_feature.png', size=(600, 600))

"""
.. figure:: vector_of_endpoints_feature.png
   :align: center

   **Showing the streamlines colored according to their orientation**.

.. include:: ../links_names.inc

.. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                    tractography simplification, Frontiers in Neuroscience, vol
                    6, no 175, 2012.
"""
