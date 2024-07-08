# code support tractometric statistical analysis for dipy
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["analysis", "qc", "resampling"],
)

__all__ = ["analysis", "qc", "resampling"]
