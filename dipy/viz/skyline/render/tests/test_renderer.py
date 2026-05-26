import numpy as np

from dipy.viz.skyline.render.renderer import (
    affine_voxel_sizes,
    slice_slider_bounds,
    slice_slider_values_from_state,
    slice_state_from_slider_values,
    voxel_values_from_slice_state,
)


def test_affine_voxel_sizes_use_affine_columns():
    """Affine voxel sizes are computed from matrix columns."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    voxel_sizes = affine_voxel_sizes(affine)

    assert np.allclose(voxel_sizes, (2.0, 3.0, 4.0))


def test_slice_slider_bounds_use_original_shape_without_affine():
    """Slice slider bounds use original sizes when no affine is provided."""
    bounds = slice_slider_bounds((10, 20, 30))

    assert bounds == ((0, 10), (0, 20), (0, 30))


def test_slice_slider_bounds_use_affine_scaled_shape_for_large_voxels():
    """Slice slider bounds use affine-scaled sizes for voxels at least one."""
    affine = np.diag([2.0, 1.0, 3.0, 1.0])

    bounds = slice_slider_bounds((10, 20, 30), affine=affine)

    assert bounds == ((0, 20), (0, 20), (0, 90))


def test_slice_slider_bounds_use_original_shape_for_small_voxels():
    """Slice slider bounds keep original sizes for voxels smaller than one."""
    affine = np.diag([0.5, 0.25, 0.9, 1.0])

    bounds = slice_slider_bounds((10, 20, 30), affine=affine)

    assert bounds == ((0, 10), (0, 20), (0, 30))


def test_slice_slider_bounds_apply_rule_per_axis():
    """Slice slider bounds apply the affine voxel-size rule independently."""
    affine = np.diag([2.0, 0.5, 1.25, 1.0])

    bounds = slice_slider_bounds((10, 20, 30), affine=affine)

    assert bounds == ((0, 20), (0, 20), (0, 38))


def test_slice_slider_values_convert_to_world_state():
    """Slice slider values convert back to affine world coordinates."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    state = slice_state_from_slider_values((8.0, 4.0, 15.0), affine=affine)

    assert np.allclose(state, (18.0, 22.0, 45.0))


def test_slice_slider_values_round_trip_from_world_state():
    """Slice slider values round-trip with affine world coordinates."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    world_state = np.array([18.0, 22.0, 45.0])

    slider_values = slice_slider_values_from_state(world_state, affine=affine)
    round_trip = slice_state_from_slider_values(slider_values, affine=affine)

    assert np.allclose(slider_values, (8.0, 4.0, 15.0))
    assert np.allclose(round_trip, world_state)


def test_slice_slider_max_value_maps_to_affine_scaled_world_extent():
    """Slider max maps to world extent for large affine voxels."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    state = slice_state_from_slider_values((20.0, 20.0, 20.0), affine=affine)

    assert np.allclose(state, (20.0, 20.0, 20.0))


def test_voxel_values_from_slice_state_inverts_affine():
    """World slice states can be converted back to voxel coordinates."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    voxel_values = voxel_values_from_slice_state((18.0, 22.0, 45.0), affine)

    assert np.allclose(voxel_values, (4.0, 4.0, 5.0))
