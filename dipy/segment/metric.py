__all__ = [
    "MinimumAverageDirectFlipMetric",
    "Metric",
    "CosineMetric",
    "AveragePointwiseEuclideanMetric",
    "EuclideanMetric",
    "dist",
    "mdf",
    "mean_manhattan_distance",
    "mean_euclidean_distance",
]

import numpy as np

from dipy.segment.metricspeed import (
    AveragePointwiseEuclideanMetric,
    CosineMetric,
    Metric,
    MinimumAverageDirectFlipMetric,
    SumPointwiseEuclideanMetric,
    dist,
)

# Creates aliases
EuclideanMetric = SumPointwiseEuclideanMetric


def mdf(s1, s2):
    """Computes the MDF (Minimum average Direct-Flip) distance between two
    streamlines.

    Streamlines must have the same number of points.

    See :footcite:p:`Garyfallidis2012a` for a definition of the distance.

    Parameters
    ----------
    s1 : 2D array
        A streamline (sequence of N-dimensional points).
    s2 : 2D array
        A streamline (sequence of N-dimensional points).

    Returns
    -------
    double
        Distance between two streamlines.

    References
    ----------
    .. footbibliography::
    """
    return dist(MinimumAverageDirectFlipMetric(), s1, s2)


def mean_manhattan_distance(a, b):
    """Compute the average Manhattan-L1 distance (MDF without flip)

    Arrays are representing a single streamline or a list of streamlines
    that have the same number of N-dimensional points (two last axis).

    Parameters
    ----------
    a : 2D or 3D array
        A streamline or concatenated streamlines
        (array of S streamlines by P points in N dimension).
    b : 2D or 3D array
        A streamline or concatenated streamlines
        (array of S streamlines by P points in N dimension).

    Returns
    -------
    1D array
        Distance between each S streamlines
    """
    return np.mean(np.sum(np.abs(a - b), axis=-1), axis=-1)


def mean_euclidean_distance(a, b):
    """Compute the average Euclidean-L2 distance (MDF without flip)

    Arrays are representing a single streamline or a list of streamlines
    that have the same number of N-dimensional points (two last axis).

    Parameters
    ----------
    a : 2D or 3D array
        A streamline or concatenated streamlines
        (array of S streamlines by P points in N dimension).
    b : 2D or 3D array
        A streamline or concatenated streamlines
        (array of S streamlines by P points in N dimension).

    Returns
    -------
    1D array
        Distance between each S streamlines
    """
    return np.mean(np.sqrt(np.sum(np.square(a - b), axis=-1)), axis=-1)
