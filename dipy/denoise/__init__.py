# init for denoise aka the denoising module
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "nlmeans",
        "non_local_means",
        "gibbs",
        "localpca",
        "patch2self",
        "adaptive_soft_matching",
        "noise_estimate",
        "shift_twist_convolution",
        "pca_noise_estimate",
        "denspeed",
        "enhancement_kernel",
        "nlmeans_block",
    ],
)

__all__ = [
    "nlmeans",
    "gibbs_removal",
    "localpca",
    "mppca",
    "p2s",
    "adaptive_soft_matching",
    "shift_twist_convolution",
    "denspeed",
    "pca_noise_estimate",
    "piesno",
    "estimate_sigma",
    "enhancement_kernel",
    "non_local_means",
    "nlmeans_block",
    "noise_estimate",
    "patch2self",
]
