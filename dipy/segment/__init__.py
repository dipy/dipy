# init for segment aka the segmentation module
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "bundles",
        "clustering_algorithms",
        "clustering",
        "clusteringspeed",
        "cythonutils",
        "featurespeed",
        "fss",
        "mask",
        "metric",
        "metricspeed",
        "mrf",
        "threshold",
        "tissue",
    ],
)

__all__ = [
    "bundles",
    "clustering_algorithms",
    "clustering",
    "clusteringspeed",
    "cythonutils",
    "featurespeed",
    "fss",
    "mask",
    "metric",
    "metricspeed",
    "mrf",
    "threshold",
    "tissue",
]
