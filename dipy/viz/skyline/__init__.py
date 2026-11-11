"""Skyline: interactive 3D viewer for diffusion images and tractography."""

from dipy.utils.optpkg import optional_package

fury_pckg_msg = (
    "Skyline requires FURY version 2.1.0 or higher. "
    "Please install or upgrade FURY using pip install -U fury."
)
fury, has_fury, _ = optional_package(
    "fury", trip_msg=fury_pckg_msg, min_version="2.1.0"
)
if has_fury:
    from dipy.viz.skyline.app import Skyline, skyline, skyline_from_files
    from dipy.viz.skyline.render.sh_slicer import create_shm_visualization

__all__ = ["Skyline", "create_shm_visualization", "skyline", "skyline_from_files"]
