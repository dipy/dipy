"""
Skyline: interactive 3D viewer for diffusion images, tractography, and related data.
"""

from dipy.utils.optpkg import optional_package

fury_pckg_msg = (
    "Skyline requires FURY version 2.0.0 or higher. "
    "Please install or upgrade FURY using pip install -U fury --pre."
)
fury, has_fury, _ = optional_package(
    "fury", trip_msg=fury_pckg_msg, min_version="2.0.0"
)
if has_fury:
    from dipy.viz.skyline.app import Skyline, skyline, skyline_from_files

__all__ = ["Skyline", "skyline", "skyline_from_files"]
