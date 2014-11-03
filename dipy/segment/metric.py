from dipy.segment.featurespeed import (Feature,
                                       IdentityFeature,
                                       MidpointFeature,
                                       CenterOfMassFeature,
                                       ArcLengthFeature)

from dipy.segment.metricspeed import (Metric,
                                      SumPointwiseEuclideanMetric,
                                      AveragePointwiseEuclideanMetric,
                                      MinimumPointwiseEuclideanMetric,
                                      MaximumPointwiseEuclideanMetric,
                                      MinimumAverageDirectFlipMetric,
                                      HausdorffMetric,
                                      ArcLengthMetric)

from dipy.segment.metricspeed import dist
# from dipy.segment.metricspeed import distance_matrix


def mdf(s1, s2):
    return dist(MinimumAverageDirectFlipMetric(), s1, s2)


# def euclidean(s1, s2, feature_type="midpoint"):
#     if feature_type == "midpoint":
#         feature_type = Midpoint()
#     elif feature_type == "center":
#         feature_type = CenterOfMass()

#     return dist(Euclidean(feature_type), s1, s2)

