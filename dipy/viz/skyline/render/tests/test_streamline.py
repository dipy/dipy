import numpy as np
import pytest

pytest.importorskip("fury")

from dipy.viz.skyline.render.streamline import create_streamline  # noqa: E402


def _minimal_polylines():
    """Two short streamlines for ``create_streamline`` tests."""
    return [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
    ]


def test_create_streamline_line_and_tube_return_actors():
    """``create_streamline`` builds line or tube geometry for ``Line`` / ``Tube``."""
    lines = _minimal_polylines()

    line_actor = create_streamline(
        lines, line_type="Line", color=np.array([1.0, 0.0, 0.0])
    )
    tube_actor = create_streamline(
        lines, line_type="Tube", color=np.array([1.0, 0.0, 0.0])
    )

    assert line_actor is not None
    assert tube_actor is not None
    assert hasattr(line_actor, "material")


def test_create_streamline_legacy_lowercase_does_not_match():
    """Lowercase ``line`` / ``tube`` are no longer valid ``line_type`` values."""
    lines = _minimal_polylines()
    assert create_streamline(lines, line_type="line") is None
    assert create_streamline(lines, line_type="tube") is None
