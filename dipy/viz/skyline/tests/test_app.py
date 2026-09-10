"""Real end-to-end tests for the Skyline viewer.

Every test drives a genuine offscreen (``stealth``) ``Skyline`` instance built
from real volumes, tractograms, peaks and surfaces. Nothing here fabricates a
stand-in for the viewer or its visualizations.
"""

import logging

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import default_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.image import save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.utils.optpkg import optional_package

_, has_pygfx, _ = optional_package("pygfx")
if has_pygfx:
    from pygfx.objects import KeyboardEvent

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.app import Skyline, skyline, skyline_from_files
    from dipy.viz.skyline.render.image import Image3D
    from dipy.viz.skyline.render.peak import Peak3D
    from dipy.viz.skyline.render.roi import ROI3D
    from dipy.viz.skyline.render.sh_slicer import SHGlyph3D
    from dipy.viz.skyline.render.streamline import ClusterStreamline3D, Streamline3D
    from dipy.viz.skyline.render.surface import Surface

AFFINE = np.eye(4)
SHAPE = (8, 9, 10)


def _volume(seed=0, shape=SHAPE):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def _image_input(name="vol.nii.gz", seed=0, shape=SHAPE):
    return (_volume(seed=seed, shape=shape), AFFINE, name)


def _roi_input(name="roi.nii.gz"):
    return ((_volume(seed=1) > 0.5).astype(np.uint8), AFFINE, name)


def _surface_input(name="surf.gii"):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    return (vertices, faces, name)


def _streamlines(n_lines=6):
    rng = np.random.default_rng(7)
    lines = []
    for index in range(n_lines):
        start = rng.random(3) * 2.0
        direction = np.array([1.0, 0.5, 0.25]) * (1 + index * 0.1)
        lines.append(np.array([start + direction * t for t in range(8)]))
    return lines


def _sft(n_lines=6):
    reference = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), AFFINE)
    return StatefulTractogram(_streamlines(n_lines), reference, Space.RASMM)


def _tractogram_input(name="tracts.trk", n_lines=6):
    return (_sft(n_lines), name)


def _pam():
    pam = PeaksAndMetrics()
    pam.affine = AFFINE
    pam.peak_dirs = np.zeros((3, 3, 3, 5, 3), dtype=np.float32)
    pam.peak_dirs[..., 0, 0] = 1.0
    pam.peak_values = np.ones((3, 3, 3, 5), dtype=np.float32)
    pam.peak_indices = np.zeros((3, 3, 3, 5), dtype=np.int32)
    pam.sphere = default_sphere
    return pam


def _peak_input(name="peaks.pam5"):
    return (_pam(), name)


def _sh_coeffs(shape=(2, 2, 2), l_max=8):
    n_coeffs = sum(2 * ell + 1 for ell in range(0, l_max + 1, 2))
    coeffs = np.zeros((*shape, n_coeffs), dtype=np.float32)
    coeffs[..., 0] = 1.0
    return coeffs


def _sh_input(name="odf"):
    return (_sh_coeffs(), AFFINE, name)


@pytest.fixture
def make_skyline(tmp_path):
    """Build real offscreen ``Skyline`` viewers that render into ``tmp_path``."""

    def _make(**kwargs):
        kwargs.setdefault("visualizer_type", "stealth")
        kwargs.setdefault("out_dir", str(tmp_path))
        return skyline(**kwargs)

    return _make


@pytest.fixture
def image_skyline(make_skyline):
    return make_skyline(images=[_image_input()])


def test_stealth_viewer_writes_the_requested_png(tmp_path, make_skyline):
    make_skyline(images=[_image_input()], out_stealth_png="shot.png")

    assert (tmp_path / "shot.png").is_file()
    assert (tmp_path / "shot.png").stat().st_size > 0


def test_stealth_viewer_defaults_to_the_dipy_skyline_filename(tmp_path, make_skyline):
    make_skyline(images=[_image_input()])

    assert (tmp_path / "DIPY SKYLINE.png").is_file()


def test_stealth_viewer_creates_a_missing_output_directory(tmp_path):
    out_dir = tmp_path / "nested" / "renders"

    skyline(
        visualizer_type="stealth",
        images=[_image_input()],
        out_dir=str(out_dir),
        out_stealth_png="shot.png",
    )

    assert (out_dir / "shot.png").is_file()


def test_stealth_viewer_has_no_sidebar(image_skyline):
    assert image_skyline.UI_window is None
    assert image_skyline._visualizer_type == "stealth"


def test_default_background_is_dark(image_skyline):
    assert image_skyline._bg_color == (0.1, 0.1, 0.1)
    assert image_skyline.window.screens[0].scene.background == (0.1, 0.1, 0.1)


def test_glass_brain_switches_the_background_to_white(make_skyline):
    viewer = make_skyline(surfaces=[_surface_input()], glass_brain=True)

    assert viewer._bg_color == (1, 1, 1)
    assert viewer.window.screens[0].scene.background == (1, 1, 1)


def test_explicit_background_color_wins_over_glass_brain(make_skyline):
    viewer = make_skyline(glass_brain=True, bg_color=(0.2, 0.4, 0.6))

    assert viewer._bg_color == (0.2, 0.4, 0.6)


def test_tract_colors_default_to_direction(image_skyline):
    assert image_skyline._tract_colors == "direction"


def test_tract_colors_string_triplet_is_parsed_into_floats(make_skyline):
    viewer = make_skyline(tract_colors="1 0 0.5")

    assert viewer._tract_colors == (1.0, 0.0, 0.5)


def test_tract_colors_named_option_is_kept_as_is(make_skyline):
    viewer = make_skyline(tract_colors="random")

    assert viewer._tract_colors == "random"


def test_every_input_type_becomes_its_own_visualization(make_skyline):
    viewer = make_skyline(
        images=[_image_input()],
        peaks=[_peak_input()],
        rois=[_roi_input()],
        surfaces=[_surface_input()],
        tractograms=[_tractogram_input()],
        sh_coeffs=[_sh_input()],
    )

    assert [type(viz) for viz in viewer.visualizations] == [
        Image3D,
        Peak3D,
        ROI3D,
        Surface,
        Streamline3D,
        SHGlyph3D,
    ]


def test_visualizations_property_concatenates_the_per_type_lists(make_skyline):
    viewer = make_skyline(
        images=[_image_input("a.nii.gz"), _image_input("b.nii.gz", seed=2)],
        rois=[_roi_input()],
    )

    assert viewer.visualizations == (
        viewer._image_visualizations + viewer._roi_visualizations
    )
    assert len(viewer._image_visualizations) == 2


def test_last_loaded_image_becomes_the_active_one(make_skyline):
    viewer = make_skyline(
        images=[_image_input("a.nii.gz"), _image_input("b.nii.gz", seed=2)]
    )

    assert viewer.active_image is viewer._image_visualizations[-1]
    assert viewer.active_image.active is True
    assert viewer.active_image.path == "b.nii.gz"


def test_viewer_without_input_has_no_visualizations(make_skyline):
    viewer = make_skyline()

    assert viewer.visualizations == []
    assert viewer.active_image is None


def test_rgb_volumes_are_loaded_as_color_images(make_skyline):
    rgb = np.random.default_rng(3).random((*SHAPE, 3)).astype(np.float32)

    viewer = make_skyline(images=[(rgb, AFFINE, "rgb.nii.gz")], rgb=True)

    assert viewer._rgb is True
    assert len(viewer._image_visualizations) == 1


def test_clustered_tractograms_build_cluster_visualizations(make_skyline):
    viewer = make_skyline(
        tractograms=[_tractogram_input(n_lines=12)], is_cluster=True, cluster_thr=2.0
    )

    assert len(viewer._tractogram_visualizations) == 1
    assert isinstance(viewer._tractogram_visualizations[0], ClusterStreamline3D)


def test_light_version_renders_streamlines_as_lines(make_skyline):
    viewer = make_skyline(tractograms=[_tractogram_input()], is_light_version=True)

    assert viewer._is_light_version is True
    assert viewer._tractogram_visualizations[0]._line_type == "Line"


@pytest.mark.parametrize("sh_coeffs", [None, []])
def test_load_visualizations_skips_empty_sh_coefficients(image_skyline, sh_coeffs):
    image_skyline._load_visualiations([], [], [], [], [], sh_coeffs)

    assert image_skyline._sh_glyph_visualizations == []


def test_load_visualizations_builds_one_glyph_per_coefficient_volume(image_skyline):
    image_skyline._load_visualiations([], [], [], [], [], [_sh_input("unit_odf")])

    assert len(image_skyline._sh_glyph_visualizations) == 1
    glyph = image_skyline._sh_glyph_visualizations[0]
    assert isinstance(glyph, SHGlyph3D)
    assert glyph.path == "unit_odf"
    assert glyph.shape == (2, 2, 2)
    assert glyph.viz_type == "sh_glyph"


def test_load_visualizations_warns_on_non_4d_sh_coefficients(image_skyline, caplog):
    flat = (np.zeros((2, 2, 45), dtype=np.float32), AFFINE, "flat_odf")

    with caplog.at_level(logging.WARNING):
        image_skyline._load_visualiations([], [], [], [], [], [flat])

    assert image_skyline._sh_glyph_visualizations == []
    assert "does not contain any SH coefficients" in caplog.text


def test_load_visualizations_warns_on_empty_tractograms(image_skyline, caplog):
    reference = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), AFFINE)
    empty = (StatefulTractogram([], reference, Space.RASMM), "empty.trk")

    with caplog.at_level(logging.WARNING):
        image_skyline._load_visualiations([], [], [], [], [empty], [])

    assert image_skyline._tractogram_visualizations == []
    assert "does not contain any streamlines" in caplog.text


def test_remove_visualization_rejects_unknown_types(image_skyline):
    with pytest.raises(ValueError, match="Unsupported visualization type"):
        image_skyline._remove_visualization(object())


def test_remove_visualization_drops_it_from_the_right_list(make_skyline):
    viewer = make_skyline(images=[_image_input()], rois=[_roi_input()])
    roi = viewer._roi_visualizations[0]

    viewer._remove_visualization(roi)

    assert viewer._roi_visualizations == []
    assert roi not in viewer.visualizations
    assert len(viewer._image_visualizations) == 1


def test_remove_visualization_clears_the_slice_focus(make_skyline):
    viewer = make_skyline(images=[_image_input()], sh_coeffs=[_sh_input()])
    glyph = viewer._sh_glyph_visualizations[0]
    viewer._slice_focus_viz = glyph

    viewer._remove_visualization(glyph)

    assert viewer._slice_focus_viz is None
    assert viewer._sh_glyph_visualizations == []


def test_remove_the_last_visualization_in_stealth_mode(image_skyline):
    image_skyline._remove_visualization(image_skyline._image_visualizations[0])

    assert image_skyline.visualizations == []


def test_enqueue_scene_op_runs_immediately_outside_the_ui_draw(image_skyline):
    calls = []
    image_skyline._refresh_requested = False

    image_skyline.enqueue_scene_op(calls.append, "now")

    assert calls == ["now"]
    assert image_skyline._pending_scene_ops == []
    assert image_skyline._refresh_requested is True


def test_enqueue_scene_op_defers_while_the_ui_is_drawing(image_skyline):
    calls = []
    image_skyline._is_drawing_ui = True

    image_skyline.enqueue_scene_op(calls.append, "later")

    assert calls == []
    assert len(image_skyline._pending_scene_ops) == 1
    assert image_skyline._refresh_requested is True


def test_enqueue_scene_op_coalesces_repeated_calls_to_one_bound_method(image_skyline):
    image = image_skyline._image_visualizations[0]
    image_skyline._is_drawing_ui = True

    image_skyline.enqueue_scene_op(image.update_state, np.array([1.0, 1.0, 1.0]))
    image_skyline.enqueue_scene_op(image.update_state, np.array([2.0, 2.0, 2.0]))

    assert len(image_skyline._pending_scene_ops) == 1
    _, args, _ = image_skyline._pending_scene_ops[0]
    npt.assert_array_equal(args[0], np.array([2.0, 2.0, 2.0]))


def test_enqueue_scene_op_keeps_same_named_methods_of_different_owners(make_skyline):
    viewer = make_skyline(
        images=[_image_input("a.nii.gz"), _image_input("b.nii.gz", seed=2)]
    )
    first, second = viewer._image_visualizations
    viewer._is_drawing_ui = True

    viewer.enqueue_scene_op(first.update_state, np.array([1.0, 1.0, 1.0]))
    viewer.enqueue_scene_op(second.update_state, np.array([2.0, 2.0, 2.0]))

    assert len(viewer._pending_scene_ops) == 2


def test_flush_pending_scene_ops_runs_and_clears_the_queue(image_skyline):
    calls = []
    image_skyline._is_drawing_ui = True
    image_skyline.enqueue_scene_op(calls.append, "first")
    image_skyline.enqueue_scene_op(lambda: calls.append("second"))
    image_skyline._is_drawing_ui = False
    image_skyline._refresh_requested = False

    image_skyline._flush_pending_scene_ops()

    assert calls == ["first", "second"]
    assert image_skyline._pending_scene_ops == []
    assert image_skyline._refresh_requested is True


def test_flush_pending_scene_ops_survives_a_failing_operation(image_skyline, caplog):
    calls = []

    def boom():
        raise RuntimeError("scene op failed")

    image_skyline._is_drawing_ui = True
    image_skyline.enqueue_scene_op(boom)
    image_skyline.enqueue_scene_op(calls.append, "still runs")
    image_skyline._is_drawing_ui = False

    with caplog.at_level(logging.ERROR):
        image_skyline._flush_pending_scene_ops()

    assert calls == ["still runs"]
    assert "Failed to apply deferred scene operation" in caplog.text


def test_flush_pending_scene_ops_is_a_noop_when_nothing_is_queued(image_skyline):
    image_skyline._refresh_requested = False

    image_skyline._flush_pending_scene_ops()

    assert image_skyline._refresh_requested is False


def test_scene_op_key_identifies_bound_methods_by_owner_and_name(make_skyline):
    viewer = make_skyline(
        images=[_image_input("a.nii.gz"), _image_input("b.nii.gz", seed=2)]
    )
    first, second = viewer._image_visualizations

    assert viewer._scene_op_key(first.update_state) == viewer._scene_op_key(
        first.update_state
    )
    assert viewer._scene_op_key(first.update_state) != viewer._scene_op_key(
        second.update_state
    )


def test_scene_op_key_falls_back_to_the_function_name(image_skyline):
    def plain_function():
        pass

    assert image_skyline._scene_op_key(plain_function) == (None, "plain_function")


def test_scene_op_key_is_none_for_anonymous_callables(image_skyline):
    class Callable:
        def __call__(self):
            pass

    assert image_skyline._scene_op_key(Callable()) is None


def test_synchronize_visualizations_pushes_the_state_to_the_other_views(make_skyline):
    viewer = make_skyline(images=[_image_input()], sh_coeffs=[_sh_input()])
    image = viewer._image_visualizations[0]
    glyph = viewer._sh_glyph_visualizations[0]
    new_state = np.array([2.0, 3.0, 4.0], dtype=np.float32)

    viewer._synchronize_visualizations(image, new_state)

    npt.assert_allclose(glyph.state, new_state)
    assert viewer._slice_focus_viz is image


def test_synchronize_visualizations_ignores_a_source_with_sync_disabled(make_skyline):
    viewer = make_skyline(images=[_image_input()], sh_coeffs=[_sh_input()])
    image = viewer._image_visualizations[0]
    glyph = viewer._sh_glyph_visualizations[0]
    before = np.array(glyph.state, dtype=float).copy()
    image._synchronize = False

    viewer._synchronize_visualizations(image, np.array([5.0, 5.0, 5.0]))

    npt.assert_allclose(glyph.state, before)
    assert viewer._slice_focus_viz is None


def test_synchronize_visualizations_queues_a_copy_while_drawing(make_skyline):
    viewer = make_skyline(images=[_image_input()], sh_coeffs=[_sh_input()])
    image = viewer._image_visualizations[0]
    glyph = viewer._sh_glyph_visualizations[0]
    before = np.array(glyph.state, dtype=float).copy()
    new_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    viewer._is_drawing_ui = True

    viewer._synchronize_visualizations(image, new_state)

    assert len(viewer._pending_sync_requests) == 1
    queued_source, queued_state = viewer._pending_sync_requests[0]
    assert queued_source is image
    npt.assert_array_equal(queued_state, new_state)
    assert queued_state is not new_state
    npt.assert_allclose(glyph.state, before)
    assert viewer._refresh_requested is True


def test_flush_pending_sync_requests_applies_the_queued_state(make_skyline):
    viewer = make_skyline(images=[_image_input()], sh_coeffs=[_sh_input()])
    image = viewer._image_visualizations[0]
    glyph = viewer._sh_glyph_visualizations[0]
    viewer._is_drawing_ui = True
    viewer._synchronize_visualizations(image, np.array([3.0, 3.0, 3.0], np.float32))
    viewer._is_drawing_ui = False

    viewer._flush_pending_sync_requests()

    assert viewer._pending_sync_requests == []
    npt.assert_allclose(glyph.state, np.array([3.0, 3.0, 3.0]))


def test_flush_pending_sync_requests_is_a_noop_when_empty(image_skyline):
    image_skyline._refresh_requested = False

    image_skyline._flush_pending_sync_requests()

    assert image_skyline._refresh_requested is False


@pytest.mark.parametrize(
    "state,expected_type",
    [
        (np.array([1.0, 2.0, 3.0]), np.ndarray),
        ([1.0, 2.0, 3.0], list),
        ((1.0, 2.0, 3.0), tuple),
    ],
)
def test_snapshot_state_copies_containers(state, expected_type):
    snapshot = Skyline._snapshot_state(state)

    assert isinstance(snapshot, expected_type)
    npt.assert_array_equal(np.asarray(snapshot), np.asarray(state))
    if not isinstance(state, tuple):
        assert snapshot is not state


def test_snapshot_state_returns_scalars_unchanged():
    assert Skyline._snapshot_state(7) == 7


def test_get_reference_slice_state_is_none_without_visualizations(make_skyline):
    viewer = make_skyline()

    assert viewer._get_reference_slice_state() is None


def test_get_reference_slice_state_prefers_the_active_image(make_skyline):
    viewer = make_skyline(
        images=[_image_input("a.nii.gz"), _image_input("b.nii.gz", seed=2)]
    )
    viewer.active_image.state = np.array([4.0, 5.0, 6.0])

    reference = viewer._get_reference_slice_state()

    npt.assert_allclose(reference, np.array([4.0, 5.0, 6.0]))


def test_get_reference_slice_state_follows_the_slice_focus(make_skyline):
    viewer = make_skyline(sh_coeffs=[_sh_input()])
    glyph = viewer._sh_glyph_visualizations[0]
    glyph.state = np.array([2.0, 3.0, 4.0])
    viewer._slice_focus_viz = glyph

    npt.assert_allclose(viewer._get_reference_slice_state(), np.array([2.0, 3.0, 4.0]))


def test_get_reference_slice_state_falls_back_to_the_last_sliceable(make_skyline):
    viewer = make_skyline(
        sh_coeffs=[_sh_input("first"), _sh_input("second")],
    )
    first, second = viewer._sh_glyph_visualizations
    first.state = np.array([0.0, 0.0, 0.0])
    second.state = np.array([7.0, 8.0, 9.0])

    npt.assert_allclose(viewer._get_reference_slice_state(), np.array([7.0, 8.0, 9.0]))


def test_get_reference_slice_state_drops_a_stale_focus(make_skyline):
    viewer = make_skyline(sh_coeffs=[_sh_input()])
    removed = viewer._sh_glyph_visualizations[0]
    viewer._slice_focus_viz = removed
    viewer._remove_visualization(removed)
    viewer._slice_focus_viz = removed

    assert viewer._get_reference_slice_state() is None
    assert viewer._slice_focus_viz is None


def test_get_reference_slice_state_ignores_surfaces_and_rois(make_skyline):
    viewer = make_skyline(rois=[_roi_input()], surfaces=[_surface_input()])

    assert viewer._get_reference_slice_state() is None


def test_apply_reference_state_only_touches_newly_loaded_views(make_skyline):
    viewer = make_skyline(images=[_image_input("old.nii.gz")])
    old = viewer._image_visualizations[0]
    old_state = np.array(old.state, dtype=float).copy()
    n_before = len(viewer._image_visualizations)
    viewer._load_visualiations([_image_input("new.nii.gz", seed=5)], [], [], [], [], [])
    new = viewer._image_visualizations[-1]
    reference = np.array([1.0, 2.0, 3.0])

    viewer._apply_reference_slice_state_to_new_visualizations(reference, n_before, 0, 0)

    npt.assert_allclose(new.state, reference)
    npt.assert_allclose(old.state, old_state)


def test_apply_reference_state_is_a_noop_without_a_reference(image_skyline):
    image = image_skyline._image_visualizations[0]
    before = np.array(image.state, dtype=float).copy()

    image_skyline._apply_reference_slice_state_to_new_visualizations(None, 0, 0, 0)

    npt.assert_allclose(image.state, before)


def test_new_visualizations_inherit_the_current_slice_position(make_skyline, tmp_path):
    viewer = make_skyline(images=[_image_input("first.nii.gz")])
    viewer.active_image.state = np.array([3.0, 4.0, 5.0])

    viewer._queue_loaded_visualizations(
        {
            "images": [_image_input("second.nii.gz", seed=9)],
            "peaks": [],
            "rois": [],
            "surfaces": [],
            "tractograms": [],
            "shm_coeffs": [],
        }
    )
    viewer._drain_pending_visualizations()

    assert len(viewer._image_visualizations) == 2
    npt.assert_allclose(viewer._image_visualizations[-1].state, [3.0, 4.0, 5.0])


def test_drain_pending_visualizations_resets_the_loading_counters(image_skyline):
    image_skyline._queue_loaded_visualizations(
        {
            "images": [],
            "peaks": [],
            "rois": [],
            "surfaces": [],
            "tractograms": [_tractogram_input()],
            "shm_coeffs": [],
        }
    )
    assert image_skyline._loading_total == 1

    image_skyline._drain_pending_visualizations()

    assert image_skyline._pending_loaded_files == []
    assert image_skyline._loading_total == 0
    assert image_skyline._loading_done == 0
    assert len(image_skyline._tractogram_visualizations) == 1


def test_update_background_color_applies_immediately(image_skyline):
    image_skyline._update_background_color((0.3, 0.6, 0.9))

    assert image_skyline._bg_color == (0.3, 0.6, 0.9)
    assert image_skyline.window.screens[0].scene.background == (0.3, 0.6, 0.9)


def test_update_background_color_is_deferred_while_drawing(image_skyline):
    image_skyline._is_drawing_ui = True

    image_skyline._update_background_color((0.5, 0.5, 0.5))

    assert image_skyline._pending_bg_color == (0.5, 0.5, 0.5)
    assert image_skyline._bg_color == (0.1, 0.1, 0.1)
    assert image_skyline._refresh_requested is True


def test_save_snapshot_writes_a_png(tmp_path, image_skyline):
    target = tmp_path / "snapshot.png"

    image_skyline._save_snapshot(str(target))

    assert target.is_file()
    assert target.stat().st_size > 0


def test_save_snapshot_appends_a_png_extension(tmp_path, image_skyline):
    image_skyline._save_snapshot(str(tmp_path / "scene"))

    assert (tmp_path / "scene.png").is_file()


def test_save_snapshot_is_deferred_while_drawing(tmp_path, image_skyline):
    target = tmp_path / "deferred.png"
    image_skyline._is_drawing_ui = True

    image_skyline._save_snapshot(str(target))

    assert not target.exists()
    assert len(image_skyline._pending_scene_ops) == 1

    image_skyline._is_drawing_ui = False
    image_skyline._flush_pending_scene_ops()

    assert target.is_file()


def test_loader_is_a_noop_without_a_sidebar(image_skyline):
    image_skyline.loader(True, message="Loading Files...")

    assert image_skyline.UI_window is None


def test_request_refresh_sets_the_refresh_flag(image_skyline):
    image_skyline._refresh_requested = False

    image_skyline.request_refresh()

    assert image_skyline._refresh_requested is True


def test_before_render_only_requests_a_refresh_while_drawing(image_skyline):
    image_skyline._is_drawing_ui = True
    image_skyline._refresh_requested = False

    image_skyline.before_render()

    assert image_skyline._refresh_requested is True


def test_before_render_clears_the_refresh_flag_when_idle(image_skyline):
    image_skyline._refresh_requested = True

    image_skyline.before_render()

    assert image_skyline._refresh_requested is False


def test_refresh_actors_keeps_the_scene_in_sync_with_the_visualizations(make_skyline):
    viewer = make_skyline(images=[_image_input()], surfaces=[_surface_input()])
    surface = viewer._surface_visualizations[0]

    assert surface.actor in viewer.window.screens[0].scene.main_scene.children

    viewer._remove_visualization(surface)
    viewer._refresh_actors()

    assert surface.actor not in viewer.window.screens[0].scene.main_scene.children


def test_update_tractogram_rendering_queues_a_mode_switch(make_skyline):
    viewer = make_skyline(tractograms=[_tractogram_input(n_lines=12)])
    tractogram = viewer._tractogram_visualizations[0]

    viewer._update_tractogram_rendering(tractogram, True)

    assert viewer._pending_tractogram_switches == [(tractogram, True)]


def test_update_tractogram_rendering_ignores_unknown_visualizations(make_skyline):
    viewer = make_skyline(tractograms=[_tractogram_input()])
    other = make_skyline(tractograms=[_tractogram_input()])._tractogram_visualizations[
        0
    ]

    viewer._update_tractogram_rendering(other, True)

    assert viewer._pending_tractogram_switches == []


def test_process_tractogram_switches_replaces_the_visualization(make_skyline):
    viewer = make_skyline(tractograms=[_tractogram_input(n_lines=12)], cluster_thr=2.0)
    tractogram = viewer._tractogram_visualizations[0]
    viewer._update_tractogram_rendering(tractogram, True)

    viewer._process_tractogram_switches()

    assert viewer._pending_tractogram_switches == []
    assert tractogram not in viewer._tractogram_visualizations
    assert viewer._loading_total == 1

    viewer._wait_for_loading_in_stealth_mode()

    assert len(viewer._tractogram_visualizations) == 1
    assert isinstance(viewer._tractogram_visualizations[0], ClusterStreamline3D)


def test_process_tractogram_switches_is_a_noop_when_nothing_is_queued(image_skyline):
    image_skyline._process_tractogram_switches()

    assert image_skyline._pending_tractogram_switches == []


@pytest.fixture
def cluster_skyline(make_skyline):
    viewer = make_skyline(
        tractograms=[_tractogram_input(n_lines=12)], is_cluster=True, cluster_thr=2.0
    )
    assert isinstance(viewer._tractogram_visualizations[0], ClusterStreamline3D)
    return viewer


def _cluster_states(cluster):
    return list(cluster._cluster_state.values())


def test_key_a_selects_every_cluster(cluster_skyline):
    cluster = cluster_skyline._tractogram_visualizations[0]

    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="a"))

    assert all(state["selected"] for state in _cluster_states(cluster))


def test_key_d_deselects_every_cluster(cluster_skyline):
    cluster = cluster_skyline._tractogram_visualizations[0]
    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="a"))

    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="d"))

    assert not any(state["selected"] for state in _cluster_states(cluster))


def test_key_e_expands_the_selected_clusters(cluster_skyline):
    cluster = cluster_skyline._tractogram_visualizations[0]
    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="a"))

    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="e"))

    states = _cluster_states(cluster)
    assert all(state["expanded"] for state in states)
    assert all(state["cluster_actor"] is not None for state in states)


def test_key_c_collapses_the_expanded_clusters(cluster_skyline):
    cluster = cluster_skyline._tractogram_visualizations[0]
    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="a"))
    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="e"))

    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="c"))

    states = _cluster_states(cluster)
    assert not any(state["expanded"] for state in states)
    assert all(state["cluster_actor"] is None for state in states)


def test_unbound_keys_leave_the_clusters_untouched(cluster_skyline):
    cluster = cluster_skyline._tractogram_visualizations[0]
    before = [dict(state) for state in _cluster_states(cluster)]

    cluster_skyline.handle_key_events(KeyboardEvent(type="key_down", key="z"))

    assert [dict(state) for state in _cluster_states(cluster)] == before


def test_handle_key_events_ignores_non_cluster_tractograms(make_skyline):
    viewer = make_skyline(tractograms=[_tractogram_input()])

    viewer.handle_key_events(KeyboardEvent(type="key_down", key="a"))

    assert isinstance(viewer._tractogram_visualizations[0], Streamline3D)


def test_skyline_from_files_loads_real_files_offscreen(tmp_path):
    volume_path = tmp_path / "vol.nii.gz"
    save_nifti(str(volume_path), _volume(), AFFINE)
    roi_path = tmp_path / "roi.nii.gz"
    save_nifti(str(roi_path), _roi_input()[0], AFFINE)
    tract_path = tmp_path / "tracts.trk"
    save_tractogram(_sft(), str(tract_path), bbox_valid_check=False)

    viewer = skyline_from_files(
        [str(volume_path), str(tract_path)],
        rois=[str(roi_path)],
        stealth=True,
        out_dir=str(tmp_path),
        out_stealth_png="from_files.png",
    )

    assert (tmp_path / "from_files.png").is_file()
    assert len(viewer._image_visualizations) == 1
    assert len(viewer._roi_visualizations) == 1
    assert len(viewer._tractogram_visualizations) == 1


def test_skyline_from_files_reports_unsupported_files(tmp_path, caplog):
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("nothing to load")

    with caplog.at_level(logging.ERROR):
        viewer = skyline_from_files(
            [str(unsupported)],
            stealth=True,
            out_dir=str(tmp_path),
            out_stealth_png="empty.png",
        )

    assert viewer.visualizations == []
    assert "is not supported in Skyline" in caplog.text
