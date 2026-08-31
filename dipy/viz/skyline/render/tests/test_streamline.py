import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.streamline import create_streamline


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


def _polylines(n_lines, *, n_points=8):
    """``n_lines`` short streamlines of ``n_points`` points each."""
    rng = np.random.default_rng(7)
    return [
        np.cumsum(rng.random((n_points, 3)), axis=0).astype(np.float32)
        for _ in range(n_lines)
    ]


@pytest.mark.parametrize("line_type", ["Line", "Tube"])
@pytest.mark.parametrize("n_lines", [2, 3, 4, 10])
def test_create_streamline_accepts_default_tuple_color(line_type, n_lines):
    """A plain RGB tuple works for any line count.

    ``len(color)`` used to be compared against ``len(lines)`` before the array
    check, so a 3-tuple raised ``AttributeError`` looking for ``.ndim``.
    """
    actor = create_streamline(_polylines(n_lines), line_type=line_type)

    assert actor is not None


@pytest.mark.parametrize("line_type", ["Line", "Tube"])
def test_create_streamline_color_forms(line_type):
    """Constant, per-line, per-point and directional colors all build actors."""
    lines = _polylines(5)
    n_points = sum(len(line) for line in lines)
    rng = np.random.default_rng(11)

    for color in (
        (1, 0, 0),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        rng.random((len(lines), 3)).astype(np.float32),
        rng.random((n_points, 3)).astype(np.float32),
        "direction",
    ):
        assert create_streamline(lines, color=color, line_type=line_type) is not None
