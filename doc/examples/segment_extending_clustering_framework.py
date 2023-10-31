"""
==========================================================
Enhancing QuickBundles with different metrics and features
==========================================================

QuickBundles [Garyfallidis12]_ is a flexible algorithm that requires only a
distance metric and an adjacency threshold to perform clustering. There is a
wide variety of metrics that could be used to cluster streamlines.

The purpose of this tutorial is to show how to easily create new ``Feature``
and new ``Metric`` classes that can be used by QuickBundles.

.. _clustering-framework:

Clustering framework
====================
DIPY_ provides a simple, flexible and fast framework to do clustering of
sequential data (e.g. streamlines).

A *sequential datum* in DIPY is represented as a numpy array of size
:math:`(N \times D)`, where each row of the array represents a $D$ dimensional
point of the sequence. A set of these sequences is represented as a list of
numpy arrays of size :math:`(N_i \times D)` for :math:`i=1:M` where $M$ is the
number of sequences in the set.

This clustering framework is modular and divided in three parts:

#. Feature extraction

#. Distance computation

#. Clustering algorithm

The **feature extraction** part includes any preprocessing needed to be done on
the data before computing distances between them (e.g. resampling the number of
points of a streamline). To define a new way of extracting features, one has to
subclass ``Feature`` (see below).

The **distance computation** part includes any metric capable of evaluating a
distance between two sets of features previously extracted from the data. To
define a new way of extracting features, one has to subclass ``Metric`` (see
below).

The **clustering algorithm** part represents the clustering algorithm itself
(e.g. QuickBundles, K-means, Hierarchical Clustering). More precisely, it
includes any algorithms taking as input a list of sequential data and
outputting a ``ClusterMap`` object.


Extending `Feature`
===================
This section will guide you through the creation of a new feature extraction
method that can be used in the context of this clustering framework. For a
list of available features in DIPY see
:ref:`sphx_glr_examples_built_segmentation_segment_clustering_features.py`.

Assuming a set of streamlines, the type of features we want to extract is the
arc length (i.e. the sum of the length of each segment for a given streamline).

Let's start by importing the necessary modules.
"""

import numpy as np

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor, colormap
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import Feature, VectorOfEndpointsFeature
from dipy.segment.metric import Metric, SumPointwiseEuclideanMetric
from dipy.tracking.streamline import length

###############################################################################
# We now define the class ``ArcLengthFeature`` that will perform the desired
# feature extraction. When subclassing ``Feature``, two methods have to be
# redefined: ``infer_shape`` and ``extract``.
#
# Also, an important property about feature extraction is whether or not
# its process is invariant to the order of the points within a streamline.
# This is needed as there is no way one can tell which extremity of a
# streamline is the beginning and which one is the end.


class ArcLengthFeature(Feature):
    """ Computes the arc length of a streamline. """
    def __init__(self):
        # The arc length stays the same even if the streamline is reversed.
        super(ArcLengthFeature, self).__init__(is_order_invariant=True)

    def infer_shape(self, streamline):
        """ Infers the shape of features extracted from `streamline`. """
        # Arc length is a scalar
        return 1

    def extract(self, streamline):
        """ Extracts features from `streamline`. """
        return length(streamline)

###############################################################################
# The new feature extraction ``ArcLengthFeature`` is ready to be used. Let's
# use it to cluster a set of streamlines by their arc length. For educational
# purposes we will try to cluster a small streamline bundle known from
# neuroanatomy as the fornix.
#
# We start by loading the fornix streamlines.

fname = get_fnames('fornix')
fornix = load_tractogram(fname, 'same',
                         bbox_valid_check=False).streamlines

streamlines = Streamlines(fornix)

###############################################################################
# Perform QuickBundles clustering using the metric
# ``SumPointwiseEuclideanMetric`` and our ``ArcLengthFeature``.

metric = SumPointwiseEuclideanMetric(feature=ArcLengthFeature())
qb = QuickBundles(threshold=2., metric=metric)
clusters = qb.cluster(streamlines)

###############################################################################
# We will now visualize the clustering result.

# Color each streamline according to the cluster they belong to.
cmap = colormap.create_colormap(np.ravel(clusters.centroids))
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, cmap):
    colormap_full[cluster.indices] = color

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colormap_full))
window.record(scene, out_path='fornix_clusters_arclength.png', size=(600, 600))

# Enables/disables interactive visualization
interactive = False
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Showing the different clusters obtained by using the arc length.
#
#
# Extending `Metric`
# ==================
# This section will guide you through the creation of a new metric that can be
# used in the context of this clustering framework. For a list of available
# metrics in DIPY see
# :ref:`sphx_glr_examples_built_segmentation_segment_clustering_metrics.py`.
#
# Assuming a set of streamlines, we want a metric that computes the cosine
# distance giving the vector between endpoints of each streamline (i.e. one
# minus the cosine of the angle between two vectors). For more information
# about this distance check
# `<https://en.wikipedia.org/wiki/Cosine_similarity>`_.
#
# We now define the class ``CosineMetric`` that will perform the desired
# distance computation. When subclassing ``Metric``, two methods have to be
# redefined: ``are_compatible`` and ``dist``. Moreover, when implementing the
# ``dist`` method, one needs to make sure the distance returned is symmetric
# (i.e. `dist(A, B) == dist(B, A)`).


class CosineMetric(Metric):
    """Compute the cosine distance between two streamlines."""
    def __init__(self):
        # For simplicity, features will be the vector between endpoints of a
        # streamline.
        super(CosineMetric, self).__init__(feature=VectorOfEndpointsFeature())

    def are_compatible(self, shape1, shape2):
        """Check if two features are vectors of same dimension.

        Basically this method exists so that we don't have to check
        inside the `dist` method (speedup).
        """
        return shape1 == shape2 and shape1[0] == 1

    def dist(self, v1, v2):
        """Compute a the cosine distance between two vectors."""
        norm = lambda x: np.sqrt(np.sum(x**2))
        cos_theta = np.dot(v1, v2.T) / (norm(v1)*norm(v2))

        # Make sure it's in [-1, 1], i.e. within domain of arccosine
        cos_theta = np.minimum(cos_theta, 1.)
        cos_theta = np.maximum(cos_theta, -1.)
        return np.arccos(cos_theta) / np.pi  # Normalized cosine distance


###############################################################################
# The new distance ``CosineMetric`` is ready to be used. Let's use
# it to cluster a set of streamlines according to the cosine distance of the
# vector between their endpoints. For educational purposes we will try to
# cluster a small streamline bundle known from neuroanatomy as the fornix.
#
# We start by loading the fornix streamlines.

fname = get_fnames('fornix')
fornix = load_tractogram(fname, 'same', bbox_valid_check=False)
streamlines = fornix.streamlines

###############################################################################
# Perform QuickBundles clustering using our metric ``CosineMetric``.

metric = CosineMetric()
qb = QuickBundles(threshold=0.1, metric=metric)
clusters = qb.cluster(streamlines)

###############################################################################
# We will now visualize the clustering result.

# Color each streamline according to the cluster they belong to.
cmap = colormap.create_colormap(np.arange(len(clusters)))
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, cmap):
    colormap_full[cluster.indices] = color

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colormap_full))
window.record(scene, out_path='fornix_clusters_cosine.png', size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Showing the different clusters obtained by using the cosine metric.
#
#
#
# References
# ----------
#
# .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
#    tractography simplification, Frontiers in Neuroscience, vol 6, no 175,
#    2012.
