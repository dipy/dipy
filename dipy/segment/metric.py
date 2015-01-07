from itertools import izip

from dipy.segment.featurespeed import (Feature,
                                       IdentityFeature)

from dipy.segment.metricspeed import (Metric,
                                      AveragePointwiseEuclideanMetric,
                                      MinimumAverageDirectFlipMetric)

from dipy.segment.metricspeed import dist
# from dipy.segment.metricspeed import distance_matrix


def mdf(s1, s2):
    if type(s1) == list and type(s2) == list:
        if len(s1) != len(s2):
            raise ValueError("Lists of streamlines must have the same length.")

        return [dist(MinimumAverageDirectFlipMetric(), e1, e2) for e1, e2 in izip(s1, s2)]

    return dist(MinimumAverageDirectFlipMetric(), s1, s2)
