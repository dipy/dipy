"""Deprecated. Polydata IO now lives beside the formats that use it.

VTK formats (.vtk, .vtp, .fib) are handled directly by :mod:`dipy.io.streamline`
for tractograms and :mod:`dipy.io.surface` for surfaces, both through polyxios.
This module only re-exports them and is removed in DIPY 2.0.
"""

import importlib

from dipy.utils.deprecator import deprecate_with_version

_NEW_LOCATIONS = {
    "convert_to_polydata": "dipy.io.stateful_surface",
    "convert_to_polydata_lines": "dipy.io.streamline",
    "get_polydata_lines": "dipy.io.streamline",
    "get_polydata_triangles": "dipy.io.surface",
    "get_polydata_vertex_attrs": "dipy.io.surface",
    "get_polydata_vertices": "dipy.io.surface",
    "load_polydata": "dipy.io.surface",
    "load_vtk_streamlines": "dipy.io.streamline",
    "save_polydata": "dipy.io.surface",
    "save_vtk_streamlines": "dipy.io.streamline",
}

__all__ = list(_NEW_LOCATIONS)


def __getattr__(name):
    """Resolve a deprecated re-export from its new location.

    Parameters
    ----------
    name : str
        Name of the attribute being looked up on this module.

    Returns
    -------
    callable
        The re-exported function, wrapped so that calling it warns.

    Raises
    ------
    AttributeError
        If `name` is not one of the deprecated re-exports.
    """
    if name not in _NEW_LOCATIONS:
        raise AttributeError(f"module 'dipy.io.vtk' has no attribute {name!r}")

    module = _NEW_LOCATIONS[name]
    return deprecate_with_version(
        f"dipy.io.vtk is deprecated. Import '{name}' from {module} instead.",
        since="1.13.0",
        until="2.0.0",
    )(getattr(importlib.import_module(module), name))
