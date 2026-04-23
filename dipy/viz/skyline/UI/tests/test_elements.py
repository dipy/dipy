import numpy as np
import pytest

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI import elements
from dipy.viz.skyline.UI.elements import colors_equal, normalize_picker_color

_, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), True),
        ((1, 0, 0), (1.0, 0.0, 0.0), True),
        (np.array([0.5, 0.25, 1.0]), (0.5, 0.25, 1.0), True),
        ((1.0, 0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 1.0), True),
        (np.array([1, 2, 3, 4]), (1, 2, 3), True),
    ],
)
def test_colors_equal_numeric_sequences(a, b, expected):
    """``colors_equal`` matches RGB regardless of alpha or container type."""
    assert colors_equal(a, b) is expected


def test_colors_equal_plain_strings():
    """``colors_equal`` compares two string operands with lexicographic equality."""
    assert colors_equal("direction", "direction") is True
    assert colors_equal("direction", "random") is False


def test_colors_equal_mixed_string_and_sequence_false():
    """``colors_equal`` is False when only one operand is a string color name."""
    assert colors_equal("direction", (1.0, 0.0, 0.0)) is False
    assert colors_equal((1.0, 0.0, 0.0), "direction") is False


def test_colors_equal_rejects_non_vector_shapes():
    """``colors_equal`` returns False for non-1D array-like inputs."""
    assert colors_equal(np.zeros((2, 3)), np.zeros((2, 3))) is False


@pytest.mark.parametrize(
    "color,fallback,expected",
    [
        ((0.25, 0.5, 0.75), None, (0.25, 0.5, 0.75)),
        (np.array([0.0, 1.0, 0.0, 0.9]), None, (0.0, 1.0, 0.0)),
        ("direction", (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)),
        ((), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0)),
        (np.array([0.1, 0.2]), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    ],
)
def test_normalize_picker_color(color, fallback, expected):
    """``normalize_picker_color`` returns RGB floats or the supplied fallback."""
    if fallback is None:
        assert normalize_picker_color(color) == expected
    else:
        assert normalize_picker_color(color, fallback=fallback) == expected


def test_normalize_picker_color_default_fallback():
    """String inputs use the default red fallback when none is passed."""
    assert normalize_picker_color("direction") == (1.0, 0.0, 0.0)


@pytest.mark.skipif(not has_imgui, reason="Requires imgui_bundle>=1.92.600")
def test_ensure_last_dir_creates_missing_directory(tmp_path):
    missing_dir = tmp_path / "new" / ".dipy"
    original_last_dir = elements._LAST_DIR
    elements._LAST_DIR = missing_dir

    try:
        resolved_dir = elements._ensure_last_dir()

        assert resolved_dir == missing_dir
        assert missing_dir.exists()
        assert missing_dir.is_dir()
    finally:
        elements._LAST_DIR = original_last_dir


@pytest.mark.skipif(not has_imgui, reason="Requires imgui_bundle>=1.92.600")
def test_ensure_last_dir_uses_parent_when_last_dir_is_file(tmp_path):
    file_path = tmp_path / "last_location.txt"
    file_path.write_text("placeholder")
    original_last_dir = elements._LAST_DIR
    elements._LAST_DIR = file_path

    try:
        resolved_dir = elements._ensure_last_dir()

        assert resolved_dir == file_path.parent
        assert resolved_dir.exists()
        assert resolved_dir.is_dir()
    finally:
        elements._LAST_DIR = original_last_dir
