import colorsys
from itertools import pairwise
import logging

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
    from fury import window
    from fury.colormap import line_colors
    from fury.lib import OrthographicCamera

    from dipy.viz.skyline.render.streamline import (
        ClusterStreamline3D,
        Streamline3D,
        _set_line_appearance,
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


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("opacity", [50, 100])
def test_line_subpixel_rendered_coverage(diagonal, opacity):
    y = 20 if diagonal else 0
    actor = create_streamline(
        [np.array([[-20.0, -y, 0], [20.0, y, 0]])],
        color=(1, 1, 1),
        opacity=opacity,
    )
    scene = window.Scene()
    scene.background = (0, 0, 0)
    scene.add(actor)
    intensities = []
    for thickness in (0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
        _set_line_appearance(actor, opacity=opacity, thickness=thickness)
        frame = window.snapshot(scene=scene, return_array=True, fname=None)
        intensities.append(int(frame[..., :3].sum()))

    assert intensities[0] > 0
    assert all(a < b for a, b in pairwise(intensities)), intensities


@pytest.mark.parametrize("thickness", [0.05, 2.0])
def test_line_opacity_rendering_is_independent_of_thickness(thickness):
    actor = create_streamline(
        [np.array([[-20.0, 0, 0], [20.0, 0, 0]])],
        color=(1, 1, 1),
        thickness=thickness,
    )
    scene = window.Scene()
    scene.background = (0, 0, 0)
    scene.add(actor)
    frames = []
    for opacity in (100, 50, 0, 100):
        _set_line_appearance(actor, opacity=opacity, thickness=thickness)
        frames.append(window.snapshot(scene=scene, return_array=True, fname=None))

    assert frames[0][..., :3].sum() > frames[1][..., :3].sum() > 0
    npt.assert_array_equal(frames[2][..., :3], 0)
    npt.assert_allclose(frames[3], frames[0], atol=1)
    npt.assert_allclose(actor.material.thickness, thickness)


def test_create_streamline_legacy_lowercase_does_not_match():
    """Lowercase ``line`` / ``tube`` are no longer valid ``line_type`` values."""
    lines = _minimal_polylines()
    assert create_streamline(lines, line_type="line") is None
    assert create_streamline(lines, line_type="tube") is None


@pytest.mark.parametrize(
    "color_form", ["rgb", "rgba", "per_line", "per_point", "direction"]
)
def test_line_world_coordinates_and_colors(color_form):
    lines = [
        np.array([[10.0, 20, 30], [20.0, 20, 30]], dtype=np.float32),
        np.array([[10.0, 25, 30], [15.0, 30, 30], [20.0, 35, 30]], dtype=np.float32),
    ]
    points = np.concatenate(lines)
    if color_form in ("rgb", "rgba"):
        color = (0.2, 0.4, 0.6) if color_form == "rgb" else (0.2, 0.4, 0.6, 0.8)
        expected_colors = np.tile(color, (len(points), 1))
    elif color_form == "per_point":
        color = np.random.default_rng(11).random((len(points), 3)).astype(np.float32)
        expected_colors = color
    else:
        color = (
            "direction"
            if color_form == "direction"
            else np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32)
        )
        per_line = line_colors(lines) if color_form == "direction" else color
        expected_colors = np.repeat(per_line, [len(p) for p in lines], axis=0)

    actor = create_streamline(lines, color=color)
    positions = actor.geometry.positions.data
    finite = np.isfinite(positions).all(axis=1)
    homogeneous = np.column_stack((positions[finite], np.ones(finite.sum())))
    world_positions = (actor.world.matrix @ homogeneous.T).T[:, :3]
    npt.assert_allclose(world_positions, points)
    npt.assert_allclose(actor.geometry.colors.data[finite], expected_colors)


@pytest.mark.parametrize("line_type", ["Line", "Tube"])
@pytest.mark.parametrize("n_lines", [2, 3, 4, 10])
def test_create_streamline_accepts_constant_color(line_type, n_lines):
    actor = create_streamline(_polylines(n_lines), line_type=line_type)
    colors = actor.geometry.colors.data
    finite = np.isfinite(colors).all(axis=1)
    npt.assert_allclose(colors[finite, :3], np.tile((1, 0, 0), (finite.sum(), 1)))


@pytest.mark.parametrize("opacity", [0, 100])
def test_line_opacity_boundaries(opacity):
    actor = create_streamline(_minimal_polylines(), opacity=opacity)
    npt.assert_allclose(actor.material.opacity, opacity / 100)


@pytest.mark.parametrize("opacity", [-1, 101, np.nan, np.inf, -np.inf])
def test_line_rejects_invalid_opacity(opacity):
    with pytest.raises(ValueError, match="^opacity must be between 0 and 100$"):
        create_streamline(_minimal_polylines(), opacity=opacity)


@pytest.mark.parametrize("thickness", [0, -0.01, np.nan, np.inf, -np.inf])
def test_line_rejects_invalid_thickness(thickness):
    with pytest.raises(
        ValueError, match="^thickness must be a positive finite number$"
    ):
        create_streamline(_minimal_polylines(), thickness=thickness)


def _assert_same_tube(actual, expected):
    """Compare tube geometry and material appearance."""
    assert type(actual.material) is type(expected.material)
    npt.assert_allclose(actual.material.opacity, expected.material.opacity)
    assert actual.material.alpha_mode == expected.material.alpha_mode
    for name in ("positions", "indices", "normals", "colors"):
        npt.assert_allclose(
            getattr(actual.geometry, name).data, getattr(expected.geometry, name).data
        )


@pytest.mark.parametrize(
    "opacity, thickness", [(-1, 0), (101, -1), (np.nan, np.inf), (np.inf, np.nan)]
)
def test_tubes_ignore_line_appearance_arguments(opacity, thickness):
    lines = _minimal_polylines()
    actual = create_streamline(
        lines, line_type="Tube", opacity=opacity, thickness=thickness
    )
    expected = create_streamline(lines, line_type="Tube")
    _assert_same_tube(actual, expected)


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


def _assert_line_appearance(actor, opacity, thickness):
    """Check retained line width and opacity."""
    npt.assert_allclose(actor.material.opacity, opacity / 100)
    npt.assert_allclose(actor.material.thickness, thickness)


def test_streamline_appearance_survives_recoloring_and_type_changes(tmp_path):
    viz = Streamline3D("bundle", _sft(_bundle()), color=(1, 1, 1))
    viz._line_opacity = 40
    viz._line_thickness = 0.03
    viz._apply_line_appearance()
    _assert_line_appearance(viz.actor, 40, 0.03)

    pvals_path = tmp_path / "pvals.npy"
    np.save(pvals_path, np.linspace(0, 1, 30))
    viz.handle_color_change([str(pvals_path)])
    _assert_line_appearance(viz.actor, 40, 0.03)
    viz._value = 0.5
    viz._update_buan_colors_on_sliders()
    _assert_line_appearance(viz.actor, 40, 0.03)

    viz._line_type = "Tube"
    viz._create_streamline_actor()
    expected_tube = create_streamline(
        viz.sft.streamlines, color=viz.color, line_type="Tube"
    )
    _assert_same_tube(viz.actor, expected_tube)
    viz._apply_line_appearance()
    _assert_same_tube(viz.actor, expected_tube)
    viz._line_type = "Line"
    viz._create_streamline_actor()
    _assert_line_appearance(viz.actor, 40, 0.03)

    scene = window.Scene()
    scene.background = (0, 0, 0)
    scene.add(viz.actor)
    partial = window.snapshot(scene=scene, return_array=True, fname=None)
    viz._line_opacity = 100
    viz._apply_line_appearance()
    full = window.snapshot(scene=scene, return_array=True, fname=None)
    viz._line_thickness = 0.10
    viz._apply_line_appearance()
    wider = window.snapshot(scene=scene, return_array=True, fname=None)
    assert 0 < partial[..., :3].sum() < full[..., :3].sum() < wider[..., :3].sum()

    viz.color = viz._original_color
    viz._line_opacity = 40
    viz._line_thickness = 0.03
    viz._create_streamline_actor()
    _assert_line_appearance(viz.actor, 40, 0.03)


def test_cluster_member_appearance_persists_without_affecting_centroids(cluster_viz):
    viz = cluster_viz
    viz._line_opacity = 40
    viz._line_thickness = 0.03
    viz._apply_line_appearance()
    for centroid in viz._cluster_state:
        npt.assert_allclose(centroid.material.opacity, 0.5)
    viz._select_all_clusters()
    viz._expand_clusters()
    for centroid, state in viz._cluster_state.items():
        _assert_line_appearance(state["cluster_actor"], 40, 0.03)
        npt.assert_allclose(centroid.material.opacity, 1)

    viz._line_opacity = 60
    viz._line_thickness = 0.07
    viz._apply_line_appearance()
    for state in viz._cluster_state.values():
        _assert_line_appearance(state["cluster_actor"], 60, 0.07)
    viz._collapse_clusters()
    viz._expand_clusters()
    for state in viz._cluster_state.values():
        _assert_line_appearance(state["cluster_actor"], 60, 0.07)

    viz._line_type = "Tube"
    viz._apply_cluster_line_type_change()
    viz._apply_line_appearance()
    for state in viz._cluster_state.values():
        expected = create_streamline(
            viz._clusters[state["cluster"]],
            color=state["color"],
            line_type="Tube",
            segments=3,
        )
        _assert_same_tube(state["cluster_actor"], expected)
    viz._line_type = "Line"
    viz._apply_cluster_line_type_change()
    for state in viz._cluster_state.values():
        _assert_line_appearance(state["cluster_actor"], 60, 0.07)

    viz._perform_clustering()
    viz._select_all_clusters()
    viz._expand_clusters()
    for state in viz._cluster_state.values():
        _assert_line_appearance(state["cluster_actor"], 60, 0.07)
    viz._deselect_all_clusters()
    for centroid in viz._cluster_state:
        npt.assert_allclose(centroid.material.opacity, 0.5)


def _depth_scene_snapshot(actors, *, opposite_view=False):
    """Render overlapping actors with a fixed orthographic camera.

    The initial frame consumes resize events before fixing the camera.

    Parameters
    ----------
    actors : list of Actor
        Actors to render against a black background.
    opposite_view : bool, optional
        View from negative rather than positive world z.

    Returns
    -------
    ndarray
        Rendered RGBA pixels.
    """
    scene = window.Scene(background=(0, 0, 0))
    scene.add(*actors)
    camera = OrthographicCamera(50, 25)
    show = window.ShowManager(
        scene=scene,
        camera=camera,
        window_type="offscreen",
        size=(800, 400),
        pixel_ratio=1,
        camera_light=False,
    )
    try:
        show.render()
        show.window.draw()
        camera.width = 50
        camera.height = 25
        camera.world.position = (0, 0, -50 if opposite_view else 50)
        camera.look_at((0, 0, 0))
        show.render()
        show.window.draw()
        return show.snapshot(fname=None)
    finally:
        show.window.close()


@pytest.mark.parametrize("thickness", [0.1, 0.3, 0.5, 2.0])
@pytest.mark.parametrize("reversed_order", [False, True])
@pytest.mark.parametrize("separate_layers", [False, True])
def test_transparent_line_nearest_color_follows_camera(
    thickness, reversed_order, separate_layers
):
    lines = [
        np.array([[-20.0, 0, 5], [20.0, 0, 5]], dtype=np.float32),
        np.array([[-20.0, 0, -5], [20.0, 0, -5]], dtype=np.float32),
    ]
    colors = np.array([[1.0, 0, 0], [0, 0, 1.0]], dtype=np.float32)
    if reversed_order:
        lines.reverse()
        colors = colors[::-1]
    if separate_layers:
        actors = [
            create_streamline([points], color=color, opacity=50, thickness=thickness)
            for points, color in zip(lines, colors)
        ]
    else:
        actors = [
            create_streamline(lines, color=colors, opacity=50, thickness=thickness)
        ]

    front_view = _depth_scene_snapshot(actors)
    rear_view = _depth_scene_snapshot(actors, opposite_view=True)
    assert front_view[..., 0].sum() > front_view[..., 2].sum()
    assert rear_view[..., 2].sum() > rear_view[..., 0].sum()


@pytest.mark.parametrize("zero_alpha_source", ["opacity", "vertex_color"])
@pytest.mark.parametrize("foreground_first", [False, True])
def test_invisible_line_does_not_occlude_other_streamlines(
    zero_alpha_source, foreground_first
):
    rear = create_streamline(
        [np.array([[-20.0, 0, -5], [20.0, 0, -5]])],
        color=(0, 0, 1),
        thickness=2,
    )
    front = create_streamline(
        [np.array([[-20.0, 0, 5], [20.0, 0, 5]])],
        color=(1, 0, 0, 0) if zero_alpha_source == "vertex_color" else (1, 0, 0),
        opacity=0 if zero_alpha_source == "opacity" else 100,
        thickness=2,
    )
    baseline = _depth_scene_snapshot([rear])
    actors = [front, rear] if foreground_first else [rear, front]
    with_invisible_front = _depth_scene_snapshot(actors)

    assert baseline[..., 2].sum() > 0
    npt.assert_allclose(with_invisible_front, baseline, atol=1)
