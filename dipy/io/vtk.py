"""Deprecated. Polydata IO now lives beside the formats that use it.

VTK formats (.vtk, .vtp, .fib) are handled directly by :mod:`dipy.io.streamline`
for tractograms and :mod:`dipy.io.surface` for surfaces, both through polyxios.
This module only re-exports them and is removed in DIPY 2.0.
"""

import warnings

from dipy.io.stateful_surface import convert_to_polydata
from dipy.io.streamline import (
    convert_to_polydata_lines,
    get_polydata_lines,
    load_vtk_streamlines,
    save_vtk_streamlines,
)
from dipy.io.surface import (
    get_polydata_triangles,
    get_polydata_vertex_attrs,
    get_polydata_vertices,
    load_polydata,
    save_polydata,
)

__all__ = [
    "convert_to_polydata",
    "convert_to_polydata_lines",
    "get_polydata_lines",
    "get_polydata_triangles",
    "get_polydata_vertex_attrs",
    "get_polydata_vertices",
    "load_polydata",
    "load_vtk_streamlines",
    "save_polydata",
    "save_vtk_streamlines",
]

warnings.warn(
    "dipy.io.vtk is deprecated and will be removed in DIPY 2.0. VTK formats "
    "(.vtk, .vtp, .fib) are handled directly by dipy.io.streamline for "
    "tractograms and dipy.io.surface for surfaces, both through polyxios. "
    "Import from those modules instead.\n"
    "* deprecated from version: 1.13\n"
    "* Will be removed as of version: 2.0",
    FutureWarning,
    stacklevel=2,
)
