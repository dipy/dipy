"""WGSL shader sources for DIPY's SH billboard rendering pipeline."""


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
    with importlib.resources.as_file(ref) as path, open(path, "rb") as fh:
        return fh.decode() if isinstance(fh, bytes) else fh.read().decode()
