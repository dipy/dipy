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
    """ Computes the MDF (Minimum average Direct-Flip) distance
    [Garyfallidis12]_ between two streamlines.

    Streamlines must have the same number of points.

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
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """
    return dist(MinimumAverageDirectFlipMetric(), s1, s2)
