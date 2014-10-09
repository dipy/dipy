from dipy.segment.metricspeed import Metric, FeatureType

from dipy.segment.metricspeed import IdentityFeature, MidpointFeature, CenterOfMassFeature
from dipy.segment.metricspeed import SumPointwiseEuclideanMetric
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.segment.metricspeed import MinimumPointwiseEuclideanMetric
from dipy.segment.metricspeed import MaximumPointwiseEuclideanMetric
from dipy.segment.metricspeed import MDF, ArcLengthMetric

from dipy.segment.metricspeed import dist
from dipy.segment.metricspeed import distance_matrix


def mdf(s1, s2):
    return dist(MDF(), s1, s2)


def euclidean(s1, s2, feature_type="midpoint"):
    if feature_type == "midpoint":
        feature_type = Midpoint()
    elif feature_type == "center":
        feature_type = CenterOfMass()

    return dist(Euclidean(feature_type), s1, s2)
