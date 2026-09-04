import numpy as np

from dipy.viz.gmem import GlobalHorizon


def test_global_horizon_window_defaults():
    gmem = GlobalHorizon()

    assert gmem.window_timer_cnt == 0


def test_global_horizon_slicer_defaults():
    gmem = GlobalHorizon()

    assert gmem.slicer_opacity == 1
    assert gmem.slicer_colormap == "gray"
    assert gmem.slicer_colormap_cnt == 0
    assert gmem.slicer_axes == ["x", "y", "z"]
    assert gmem.slicer_rgb is False
    assert gmem.slicer_grid is False


def test_global_horizon_first_colormap_matches_the_default():
    gmem = GlobalHorizon()

    assert gmem.slicer_colormaps[gmem.slicer_colormap_cnt] == gmem.slicer_colormap
    assert gmem.slicer_colormaps == [
        "gray",
        "magma",
        "viridis",
        "jet",
        "Pastel1",
        "disting",
    ]


def test_global_horizon_slicer_state_starts_empty():
    gmem = GlobalHorizon()

    unset = [
        gmem.slicer_curr_x,
        gmem.slicer_curr_y,
        gmem.slicer_curr_z,
        gmem.slicer_curr_actor_x,
        gmem.slicer_curr_actor_y,
        gmem.slicer_curr_actor_z,
        gmem.slicer_orig_shape,
        gmem.slicer_resliced_shape,
        gmem.slicer_vol_idx,
        gmem.slicer_vol,
        gmem.slicer_peaks_actor_z,
    ]

    assert all(value is None for value in unset)


def test_global_horizon_tractogram_defaults():
    gmem = GlobalHorizon()

    assert gmem.cluster_thr == 15
    assert gmem.streamline_actors == []
    assert gmem.centroid_actors == []
    assert gmem.cluster_actors == []


def test_global_horizon_instances_do_not_share_mutable_state():
    first = GlobalHorizon()
    second = GlobalHorizon()

    first.streamline_actors.append("actor")
    first.slicer_colormaps.append("plasma")

    assert second.streamline_actors == []
    assert "plasma" not in second.slicer_colormaps


def test_global_horizon_holds_arbitrary_slicer_state():
    gmem = GlobalHorizon()
    volume = np.zeros((4, 5, 6))

    gmem.slicer_vol = volume
    gmem.slicer_vol_idx = 2
    gmem.slicer_orig_shape = volume.shape
    gmem.slicer_curr_x = 1

    assert gmem.slicer_vol is volume
    assert gmem.slicer_vol_idx == 2
    assert gmem.slicer_orig_shape == (4, 5, 6)
    assert gmem.slicer_curr_x == 1
