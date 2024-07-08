# init for nn aka the deep neural network module
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "cnn_1d_denoising",
        "deepn4",
        "evac",
        "histo_resdnn",
        "model",
        "synb0",
        "utils",
    ],
)

__all__ = [
    "cnn_1d_denoising",
    "deepn4",
    "evac",
    "histo_resdnn",
    "model",
    "synb0",
    "utils",
]
