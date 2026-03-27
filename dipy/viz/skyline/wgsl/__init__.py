"""WGSL shader sources for DIPY's SH billboard rendering pipeline."""


def _register_dipy_wgsl_loader():
    """Register this package with Fury's Jinja WGSL loader if dependencies exist.
    None
        Failures are swallowed so Skyline degrades gracefully without WGSL extras.
    """
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
    """Load a UTF-8 WGSL/Jinja template shipped inside ``dipy.viz.skyline.wgsl``.

    Parameters
    ----------
    name : str
        Relative filename inside the package resource tree.

    Returns
    -------
    str
        Shader source text.
    """
    import importlib.resources

    ref = importlib.resources.files(__package__) / name
    with importlib.resources.as_file(ref) as path:
        with open(path, "rb") as fh:
            return fh.decode() if isinstance(fh, bytes) else fh.read().decode()


_register_dipy_wgsl_loader()
