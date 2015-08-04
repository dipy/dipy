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
All examples assume a variable `streamlines` has already been loaded. We
defined a simple function to do so. It imports the necessary modules and
load a small streamline bundle.
"""


def get_streamlines():
    from nibabel import trackvis as tv
    from dipy.data import get_data

    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    return streamlines

"""
.. _clustering-examples-SumPointwiseEuclideanMetric:

Sum of Pointwise Euclidean Metric
=================================
TODO
"""
