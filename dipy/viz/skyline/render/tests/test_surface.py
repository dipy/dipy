import numpy as np
import pytest

pytest.importorskip("fury")

from dipy.viz.skyline.render.surface import Surface, create_surface_visualization  # noqa: E402


def _triangle_mesh():
    """Minimal single-triangle mesh for ``surface()`` tests."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    return vertices, faces


def test_create_surface_visualization_rejects_invalid_input():
    """``create_surface_visualization`` raises when ``input`` is not a 2- or 3-tuple."""
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_surface_visualization("bad", 0)
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_surface_visualization((np.zeros(3),), 0)


def test_create_surface_visualization_two_tuple_names_surface_by_index():
    """A two-tuple ``(vertices, faces)`` uses the default ``Surface_{idx}`` filename."""
    vertices, faces = _triangle_mesh()
    viz = create_surface_visualization(
        (vertices, faces), idx=5, color=(0.1, 0.2, 0.3)
    )
    assert viz.path == "Surface_5"
    assert viz._color_picker_popup_id == "surface_color_picker_popup##Surface_5"


def test_create_surface_visualization_three_tuple_uses_filename():
    """A three-tuple carries an explicit filename as the third element."""
    vertices, faces = _triangle_mesh()
    viz = create_surface_visualization((vertices, faces, "pial.gii"), idx=0)
    assert viz.path == "pial.gii"


def test_surface_color_picker_initial_state():
    """Draft color and popup id match the constructor color and name."""
    vertices, faces = _triangle_mesh()
    color = (0.5, 0.0, 1.0)
    viz = Surface("hemi", vertices, faces, color=color, render_callback=None)
    assert viz._draft_color == color
    assert viz._color_picker_open is False
    assert viz._color_picker_popup_id == "surface_color_picker_popup##hemi"


def test_surface_populate_info_and_actor():
    """Info lists vertex and face counts; ``actor`` is the Fury surface actor."""
    vertices, faces = _triangle_mesh()
    viz = Surface("surf", vertices, faces, render_callback=None)

    info = viz._populate_info()
    assert "No. of vertices: 3" in info
    assert "No. of faces: 1" in info
    assert viz.actor is viz._surface_actor
