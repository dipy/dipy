"""WGSL shader sources for DIPY's SH billboard rendering pipeline."""

from __future__ import annotations


def _register_dipy_wgsl_loader():
    try:
        import jinja2
        from pygfx.renderers.wgpu.shader.templating import (
            register_wgsl_loader,
            root_loader,
        )

        if "dipy" not in root_loader.mapping:
            register_wgsl_loader(
                "dipy",
                jinja2.PackageLoader("dipy.viz.wgsl", "."),
            )
    except Exception:
        pass


def load_dipy_wgsl(name: str) -> str:
    import importlib.resources

    ref = importlib.resources.files(__package__) / name
    with importlib.resources.as_file(ref) as path:
        with open(path, "rb") as fh:
            return fh.decode() if isinstance(fh, bytes) else fh.read().decode()


_register_dipy_wgsl_loader()
