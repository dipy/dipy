# init for simulations
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["phantom", "voxel"],
)

__all__ = ["phantom", "voxel"]
