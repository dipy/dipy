from dipy.segment.featurespeed import (Feature,
                                       IdentityFeature,
                                       CenterOfMassFeature)

from dipy.segment.metricspeed import (Metric,
                                      SumPointwiseEuclideanMetric,
                                      AveragePointwiseEuclideanMetric,
                                      MinimumAverageDirectFlipMetric)

from dipy.segment.metricspeed import (dist,
                                      distance_matrix)


def mdf(s1, s2):
    """ Computes the MDF (Minimum average Direct-Flip) distance between two
    streamlines.

    Streamlines must have the same number of points.

    Parameters
    ----------
    s1 : 2D array
        a streamline (sequence of N-dimensional points)
    s2 : 2D array
        a streamline (sequence of N-dimensional points)

    Returns
    -------
    distance : double
        distance between two streamlines
    """
    return dist(MinimumAverageDirectFlipMetric(), s1, s2)
