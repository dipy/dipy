import numpy as np
import pytest

from dipy.viz.skyline.UI.elements import colors_equal, normalize_picker_color


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
