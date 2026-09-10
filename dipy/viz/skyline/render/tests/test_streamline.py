import colorsys

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.streamline import (
        ClusterStreamline3D,
        Streamline3D,
        apply_buan_colors,
        create_cluster_help,
        create_colormap,
        create_streamline,
        create_streamline_visualization,
    )

AFFINE = np.eye(4)
SHAPE = (16, 16, 16)


def _minimal_polylines():
    """Two short streamlines for ``create_streamline`` tests."""
    return [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
    ]


def _polylines(n_lines, *, n_points=8):
    """``n_lines`` short streamlines of ``n_points`` points each."""
    rng = np.random.default_rng(7)
    return Streamlines(
        [
            np.cumsum(rng.random((n_points, 3)), axis=0).astype(np.float32)
            for _ in range(n_lines)
        ]
    )


def _bundle(n_lines=24, n_points=12):
    """Two well-separated fibre groups, so clustering finds real clusters."""
    rng = np.random.default_rng(3)
    lines = []
    for index in range(n_lines):
        offset = np.array([0.0, 0.0, 0.0]) if index % 2 else np.array([10.0, 0.0, 0.0])
        jitter = rng.random(3) * 0.4
        lines.append(
            np.array(
                [offset + jitter + np.array([0.0, t, 0.0]) for t in range(n_points)],
                dtype=np.float32,
            )
        )
    return lines


def _sft(lines=None):
    reference = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), AFFINE)
    return StatefulTractogram(
        lines if lines is not None else _polylines(6), reference, Space.RASMM
    )


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


def test_create_colormap_shape_and_range():
    lut = create_colormap(16)

    assert lut.shape == (16, 3)
    assert lut.dtype == np.float32
    assert lut.min() >= 0.0
    assert lut.max() <= 1.0


def test_create_colormap_interpolates_between_the_endpoints():
    lut = create_colormap(5, hue=(0.0, 1.0), saturation=(1.0, 0.0), value=0.5)

    npt.assert_allclose(lut[0], colorsys.hsv_to_rgb(0.0, 1.0, 0.5), atol=1e-6)
    npt.assert_allclose(lut[-1], colorsys.hsv_to_rgb(1.0, 0.0, 0.5), atol=1e-6)
    npt.assert_allclose(lut[2], colorsys.hsv_to_rgb(0.5, 0.5, 0.5), atol=1e-6)


def test_create_colormap_constant_value_channel():
    lut = create_colormap(8, hue=(0.0, 0.0), saturation=(0.0, 0.0), value=0.3)

    npt.assert_allclose(lut, np.full((8, 3), 0.3), atol=1e-6)


def test_apply_buan_colors_returns_one_color_per_point():
    lines = _polylines(4, n_points=6)
    pvals = np.linspace(0.0, 1.0, 20)

    colors, color_idx = apply_buan_colors(lines, pvals)

    n_points = sum(len(line) for line in lines)
    assert colors.shape == (n_points, 3)
    assert color_idx.shape == (n_points,)
    assert color_idx.min() >= 0
    assert color_idx.max() <= len(pvals) - 1


def test_apply_buan_colors_reuses_precomputed_indices():
    lines = _polylines(4, n_points=6)
    pvals = np.linspace(0.0, 1.0, 20)
    _, color_idx = apply_buan_colors(lines, pvals)

    colors, reused_idx = apply_buan_colors(lines, pvals, buan_color_idx=color_idx)

    npt.assert_array_equal(reused_idx, color_idx)
    npt.assert_allclose(colors, create_colormap(len(pvals))[color_idx])


def test_apply_buan_colors_follows_the_hue_and_value_settings():
    lines = _polylines(3, n_points=5)
    pvals = np.linspace(0.0, 1.0, 10)
    _, color_idx = apply_buan_colors(lines, pvals)

    colors, _ = apply_buan_colors(
        lines,
        pvals,
        buan_color_idx=color_idx,
        hue=(0.5, 0.5),
        saturation=(0.0, 0.0),
        value=0.25,
    )

    npt.assert_allclose(colors, np.full(colors.shape, 0.25), atol=1e-6)


def test_apply_buan_colors_caps_the_band_count(caplog):
    import logging

    lines = _polylines(3, n_points=5)
    pvals = np.linspace(0.0, 1.0, 1500)

    with caplog.at_level(logging.INFO):
        colors, color_idx = apply_buan_colors(lines, pvals)

    assert color_idx.max() <= 999
    assert colors.shape[1] == 3
    assert "Limiting assignment to 1000 bands" in caplog.text


def test_create_cluster_help_lists_every_shortcut():
    help_block = create_cluster_help(position=(10, 20), size=(220, 190))

    for shortcut in ("'e' to expand", "'c' to collapse", "'a' to select all"):
        assert shortcut in help_block.message
    assert help_block is not None


@pytest.mark.parametrize("bad_input", ["not a tuple", (), (1, 2, 3)])
def test_create_streamline_visualization_rejects_invalid_input(bad_input):
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_streamline_visualization(bad_input, 0)


def test_create_streamline_visualization_names_by_index():
    viz = create_streamline_visualization((_sft(),), 3)

    assert viz.path == "Streamline_3"
    assert isinstance(viz, Streamline3D)


def test_create_streamline_visualization_uses_the_given_filename():
    viz = create_streamline_visualization((_sft(), "af_left.trk"), 0)

    assert viz.path == "af_left.trk"


def test_create_streamline_visualization_direction_coloring():
    viz = create_streamline_visualization(
        (_sft(), "t.trk"), 0, tract_colors="direction"
    )

    assert viz.color == "direction"


def test_create_streamline_visualization_random_color_from_the_colormap():
    colors = iter([(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)])

    viz = create_streamline_visualization(
        (_sft(), "t.trk"), 0, tract_colors="random", colormap=colors
    )

    assert viz.color == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("color", [(1, 0, 0), (1, 0, 0, 0.5)])
def test_create_streamline_visualization_explicit_color(color):
    viz = create_streamline_visualization((_sft(), "t.trk"), 0, tract_colors=color)

    assert viz.color == color


def test_create_streamline_visualization_rejects_an_unknown_color_option():
    with pytest.raises(ValueError, match="Invalid tract_colors value"):
        create_streamline_visualization((_sft(), "t.trk"), 0, tract_colors="rainbow")


def test_create_streamline_visualization_builds_a_cluster_view():
    viz = create_streamline_visualization(
        (_sft(_bundle()), "t.trk"),
        0,
        is_cluster=True,
        thr=2.0,
        async_clustering=False,
    )

    assert isinstance(viz, ClusterStreamline3D)
    assert viz.thr == 2.0


def test_streamline3d_reports_its_streamline_counts_and_lengths():
    viz = Streamline3D("bundle", _sft(_polylines(5, n_points=10)))

    info = viz._populate_info()

    assert "Number of streamlines: 5" in info
    assert "Min Length:" in info
    assert "Max Length:" in info


def test_streamline3d_exposes_a_fury_actor():
    viz = Streamline3D("bundle", _sft())

    assert viz.actor is viz._actor
    assert hasattr(viz.actor, "material")


def test_streamline3d_defaults():
    viz = Streamline3D("bundle", _sft(), color=(0.2, 0.4, 0.6))

    assert viz.color == (0.2, 0.4, 0.6)
    assert viz._original_color == (0.2, 0.4, 0.6)
    assert viz._draft_color == (0.2, 0.4, 0.6)
    assert viz._color_picker_open is False
    assert viz._color_picker_popup_id == "streamline_color_picker_popup##bundle"
    assert viz._line_type == "Line"
    assert viz.viz_type == "tractography"


def test_streamline3d_tube_rendering():
    viz = Streamline3D("bundle", _sft(), line_type="Tube")

    assert viz._line_type == "Tube"
    assert viz.actor is not None


def test_streamline3d_applies_buan_colors_from_a_file(tmp_path):
    sft = _sft(_polylines(5, n_points=10))
    pvals_path = tmp_path / "pvals.npy"
    np.save(str(pvals_path), np.linspace(0.0, 1.0, 30))
    viz = Streamline3D("bundle", sft)

    viz.handle_color_change([str(pvals_path)])

    assert viz._buan_pvals_file == "pvals.npy"
    assert viz._buan_pvals_data.shape == (30,)
    assert viz._buan_color_idx is not None
    assert viz.color.shape[1] == 3


def test_streamline3d_loads_buan_colors_at_construction(tmp_path):
    pvals_path = tmp_path / "pvals.npy"
    np.save(str(pvals_path), np.linspace(0.0, 1.0, 30))

    viz = Streamline3D(
        "bundle",
        _sft(_polylines(5, n_points=10)),
        buan_pvals_file=[str(pvals_path)],
    )

    assert viz._buan_pvals_file == "pvals.npy"
    assert viz.color.shape[1] == 3


def test_streamline3d_ignores_a_missing_buan_file():
    viz = Streamline3D("bundle", _sft(), color=(1, 0, 0))

    viz.handle_color_change(None)

    assert viz.color == (1, 0, 0)
    assert viz._buan_color_idx is None


def test_streamline3d_slider_updates_reuse_the_buan_indices(tmp_path):
    pvals_path = tmp_path / "pvals.npy"
    np.save(str(pvals_path), np.linspace(0.0, 1.0, 30))
    viz = Streamline3D("bundle", _sft(_polylines(5, n_points=10)))
    viz.handle_color_change([str(pvals_path)])
    original_idx = viz._buan_color_idx.copy()

    viz._value = 0.25
    viz._hue_low = 0.5
    viz._hue_high = 0.5
    viz._saturation_high = 0.0
    viz._saturation_low = 0.0
    viz._update_buan_colors_on_sliders()

    npt.assert_array_equal(viz._buan_color_idx, original_idx)
    npt.assert_allclose(viz.color, np.full(viz.color.shape, 0.25), atol=1e-6)


@pytest.fixture
def cluster_viz():
    return ClusterStreamline3D(
        "bundle", _sft(_bundle()), 2.0, async_clustering=False, size_threshold=1
    )


def test_cluster_streamline_clusters_the_input(cluster_viz):
    assert len(cluster_viz._clusters) >= 2
    assert len(cluster_viz._cluster_state) == len(cluster_viz._clusters)
    assert cluster_viz._sizes.sum() == len(cluster_viz.sft.streamlines)
    assert cluster_viz.viz_type == "tractography"


def test_cluster_streamline_thresholds_default(cluster_viz):
    assert cluster_viz.thr == 2.0
    assert cluster_viz.size == 1
    assert cluster_viz.length == 20.0


def test_cluster_streamline_uses_the_documented_fallback_thresholds():
    viz = ClusterStreamline3D("bundle", _sft(_bundle()), 2.0, async_clustering=False)

    assert viz.size == 10
    assert viz.length == 20.0


def test_cluster_streamline_reports_cluster_statistics(cluster_viz):
    info = cluster_viz._populate_info()

    assert f"Total streamlines: {len(cluster_viz.sft.streamlines)}" in info
    assert f"Number of clusters: {len(cluster_viz._clusters)}" in info
    assert "Max Cluster Size:" in info
    assert "Min Cluster Length:" in info


def test_cluster_streamline_select_and_deselect_every_cluster(cluster_viz):
    cluster_viz._select_all_clusters()
    assert all(s["selected"] for s in cluster_viz._cluster_state.values())

    cluster_viz._deselect_all_clusters()
    assert not any(s["selected"] for s in cluster_viz._cluster_state.values())


def test_cluster_streamline_toggle_flips_one_cluster(cluster_viz):
    centroid = next(iter(cluster_viz._cluster_state))
    before = cluster_viz._cluster_state[centroid]["selected"]

    cluster_viz._toggle_cluster_selection(centroid)

    assert cluster_viz._cluster_state[centroid]["selected"] is not before


def test_cluster_streamline_expand_then_collapse(cluster_viz):
    cluster_viz._select_all_clusters()

    cluster_viz._expand_clusters()
    assert all(s["expanded"] for s in cluster_viz._cluster_state.values())
    assert all(
        s["cluster_actor"] is not None for s in cluster_viz._cluster_state.values()
    )

    cluster_viz._collapse_clusters()
    assert not any(s["expanded"] for s in cluster_viz._cluster_state.values())
    assert all(s["cluster_actor"] is None for s in cluster_viz._cluster_state.values())


def test_cluster_streamline_expand_is_a_noop_without_a_selection(cluster_viz):
    cluster_viz._deselect_all_clusters()

    cluster_viz._expand_clusters()

    assert not any(s["expanded"] for s in cluster_viz._cluster_state.values())


def test_cluster_streamline_hide_and_show(cluster_viz):
    cluster_viz._deselect_all_clusters()

    cluster_viz._hide_deselected_clusters()
    assert not any(centroid.visible for centroid in cluster_viz._cluster_state)

    cluster_viz._show_all_clusters()
    assert all(centroid.visible for centroid in cluster_viz._cluster_state)


def test_cluster_streamline_show_and_refresh_reapplies_the_thresholds(cluster_viz):
    cluster_viz._hide_deselected_clusters()
    cluster_viz.size = 10**6

    cluster_viz._show_all_clusters_and_refresh()

    assert not any(centroid.visible for centroid in cluster_viz._cluster_state)

    cluster_viz.size = 0
    cluster_viz.length = 0.0
    cluster_viz._show_all_clusters_and_refresh()

    assert all(centroid.visible for centroid in cluster_viz._cluster_state)


def test_cluster_streamline_line_type_change_rebuilds_expanded_actors(cluster_viz):
    cluster_viz._select_all_clusters()
    cluster_viz._expand_clusters()
    before = [s["cluster_actor"] for s in cluster_viz._cluster_state.values()]

    cluster_viz._line_type = "Tube"
    cluster_viz._apply_cluster_line_type_change()

    after = [s["cluster_actor"] for s in cluster_viz._cluster_state.values()]
    assert all(new is not None for new in after)
    assert all(new is not old for new, old in zip(after, before))


def test_cluster_streamline_visible_tractogram_follows_the_selection(cluster_viz):
    cluster_viz._select_all_clusters()
    all_selected = cluster_viz.compute_visible_tractogram()

    cluster_viz._deselect_all_clusters()
    none_selected = cluster_viz.compute_visible_tractogram()

    assert len(all_selected.streamlines) == len(cluster_viz.sft.streamlines)
    assert len(none_selected.streamlines) == 0


def test_cluster_streamline_saves_the_visible_tractogram(tmp_path, cluster_viz):
    target = tmp_path / "visible.trk"
    cluster_viz._select_all_clusters()

    cluster_viz.save_tractogram([str(target)])

    assert target.is_file()
    saved = load_tractogram(str(target), "same", bbox_valid_check=False)
    assert len(saved.streamlines) == len(cluster_viz.sft.streamlines)


def test_cluster_streamline_save_accepts_a_plain_path(tmp_path, cluster_viz):
    target = tmp_path / "plain.trk"
    cluster_viz._select_all_clusters()

    cluster_viz.save_tractogram(str(target))

    assert target.is_file()


def test_cluster_streamline_save_ignores_an_empty_selection_of_files(
    tmp_path, cluster_viz
):
    cluster_viz.save_tractogram(None)
    cluster_viz.save_tractogram([])

    assert list(tmp_path.iterdir()) == []


def test_cluster_streamline_exposes_a_group_actor(cluster_viz):
    assert cluster_viz.actor is cluster_viz._actor
    assert len(cluster_viz.actor.children) >= 1
