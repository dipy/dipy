"""Deprecated location of the DeepN4 model, which moved to
:mod:`dipy.nn.deepn4`."""

from dipy.nn.deepn4 import DeepN4
from dipy.nn.torch.unet3d import UNet3D
from dipy.utils.deprecator import deprecate_with_version

__all__ = ["DeepN4", "UNet3D"]

_DEPRECATION = (
    "`dipy.nn.torch.deepn4` is deprecated: DeepN4 no longer needs PyTorch. "
    "Use `dipy.nn.deepn4.DeepN4` instead, and `dipy.nn.torch.unet3d.UNet3D` "
    "for the PyTorch network."
)

deprecate_with_version(_DEPRECATION, since="1.13", until="1.15")(lambda: None)()
