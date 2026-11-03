import colorsys

import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package
from dipy.viz.colormap import (
    create_colormap,
    map_scalars_to_lut,
    scalars_to_lut_indices,
)

_, has_fury, _ = optional_package("fury", min_version="2.1.0")
if has_fury:
    from fury.colormap import create_colormap as fury_create_colormap

LUT = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)

needs_fury = pytest.mark.skipif(not has_fury, reason="Requires fury>=2.1.0")


@needs_fury
def test_create_colormap_named_colormap_matches_fury():
    values = np.linspace(0, 1, 7)

    npt.assert_array_equal(
        create_colormap(values, name="viridis"),
        fury_create_colormap(values, name="viridis"),
    )


@needs_fury
def test_create_colormap_hsv_range_interpolates_between_the_endpoints():
    lut = create_colormap(
        np.linspace(0, 1, 5),
        auto=False,
        hsv_range=((0.0, 1.0), (1.0, 0.0), (0.5, 0.5)),
    )

    assert lut.shape == (5, 3)
    npt.assert_allclose(lut[0], colorsys.hsv_to_rgb(0.0, 1.0, 0.5), atol=1e-6)
    npt.assert_allclose(lut[-1], colorsys.hsv_to_rgb(1.0, 0.0, 0.5), atol=1e-6)
    npt.assert_allclose(lut[2], colorsys.hsv_to_rgb(0.5, 0.5, 0.5), atol=1e-6)


@needs_fury
def test_create_colormap_hsv_range_constant_value_channel():
    lut = create_colormap(
        np.linspace(0, 1, 8),
        auto=False,
        hsv_range=((0.0, 0.0), (0.0, 0.0), (0.3, 0.3)),
    )

    npt.assert_allclose(lut, np.full((8, 3), 0.3), atol=1e-6)


@needs_fury
def test_create_colormap_hsv_range_auto_rescales_and_clips():
    hsv_range = ((0.0, 0.5), (1.0, 1.0), (1.0, 1.0))
    rescaled = create_colormap([10.0, 20.0, 30.0], hsv_range=hsv_range)
    clipped = create_colormap([-1.0, 0.5, 2.0], auto=False, hsv_range=hsv_range)

    npt.assert_allclose(rescaled, clipped, atol=1e-6)


@needs_fury
def test_create_colormap_rejects_incomplete_hsv_range():
    with pytest.raises(ValueError, match="hsv_range"):
        create_colormap(np.linspace(0, 1, 4), hsv_range=((0.0, 1.0), (0.0, 1.0)))


def test_indices_span_the_table_between_the_clim_endpoints():
    indices = scalars_to_lut_indices([0.0, 0.25, 0.5, 0.75, 1.0], (0.0, 1.0), 5)

    npt.assert_array_equal(indices, [0, 1, 2, 3, 4])


def test_indices_round_to_the_nearest_entry():
    indices = scalars_to_lut_indices([0.1, 0.4, 0.6, 0.9], (0.0, 1.0), 3)

    npt.assert_array_equal(indices, [0, 1, 1, 2])


def test_indices_clip_out_of_range_and_infinite_values():
    indices = scalars_to_lut_indices([-5.0, -np.inf, 5.0, np.inf], (0.0, 1.0), 4)

    npt.assert_array_equal(indices, [0, 0, 3, 3])


def test_indices_keep_the_input_shape():
    values = np.linspace(0, 1, 6).reshape(2, 3)

    assert scalars_to_lut_indices(values, (0.0, 1.0), 4).shape == (2, 3)


def test_indices_of_empty_input_are_empty():
    assert scalars_to_lut_indices([], (0.0, 1.0), 4).shape == (0,)


def test_indices_of_zero_width_clim_select_the_first_entry():
    indices = scalars_to_lut_indices([-1.0, 2.0, 2.0], (2.0, 2.0), 4)

    npt.assert_array_equal(indices, [0, 0, 0])


def test_indices_of_a_single_color_table_are_zero():
    npt.assert_array_equal(scalars_to_lut_indices([0.0, 1.0], (0.0, 1.0), 1), [0, 0])


@pytest.mark.parametrize(
    "clim", [(1.0, 0.0), (np.nan, 1.0), (0.0, np.nan), (0.0, np.inf), (-np.inf, 0.0)]
)
def test_indices_reject_invalid_clim(clim):
    with pytest.raises(ValueError, match="clim"):
        scalars_to_lut_indices([0.5], clim, 4)


def test_indices_reject_nan_values():
    with pytest.raises(ValueError, match="values"):
        scalars_to_lut_indices([0.5, np.nan], (0.0, 1.0), 4)


def test_indices_reject_empty_table():
    with pytest.raises(ValueError, match="n_colors"):
        scalars_to_lut_indices([0.5], (0.0, 1.0), 0)


def test_map_scalars_to_lut_returns_the_table_colors():
    colors = map_scalars_to_lut([10.0, 15.0, 20.0, 99.0], (10.0, 20.0), LUT)

    npt.assert_array_equal(colors, LUT[[0, 2, 4, 4]])


def test_map_scalars_to_lut_appends_a_color_axis():
    colors = map_scalars_to_lut(np.zeros((2, 3)), (0.0, 1.0), LUT)

    assert colors.shape == (2, 3, 3)
