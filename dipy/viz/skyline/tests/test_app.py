import numpy as np
import pytest

from dipy.viz.skyline.app import Skyline
from dipy.viz.skyline.render.image import Image3D
from dipy.viz.skyline.render.sh_slicer import SHGlyph3D

pytest.importorskip("fury")


def _skyline_stub_for_load_visualizations():
    """Build a ``Skyline`` instance without running ``__init__`` for loading tests.

    Returns
    -------
    Skyline
        Instance with the attributes ``_load_visualiations`` reads from ``self``.
    """
    obj = Skyline(visualizer_type="stealth")
    obj.UI_window = None
    obj._rgb = False
    obj._glass_brain = False
    obj._is_cluster = False
    obj._cluster_thr = 15.0
    obj._is_light_version = False
    obj._tract_colors = "direction"
    obj._cluster_size_thr = None
    obj._cluster_length_thr = None
    obj._buan_pvals = None
    obj._direct_load = False
    obj._visualizer_type = "stealth"
    obj._color_gen = iter(())
    obj._image_visualizations = []
    obj._peak_visualizations = []
    obj._roi_visualizations = []
    obj._surface_visualizations = []
    obj._tractogram_visualizations = []
    obj._sh_glyph_visualizations = []
    obj.request_refresh = lambda: None
    obj._synchronize_visualizations = lambda *args, **kwargs: None
    obj._update_tractogram_rendering = lambda *args, **kwargs: None
    obj.loader = lambda *args, **kwargs: None
    return obj


@pytest.mark.parametrize("sh_coeffs", [None, []])
def test_load_visualizations_empty_sh_coeffs(sh_coeffs):
    """``_load_visualiations`` skips SH glyphs when ``sh_coeffs`` is empty.

    Parameters
    ----------
    sh_coeffs : list or None
        Spherical harmonics inputs passed to ``_load_visualiations`` (empty).
    """
    sky = _skyline_stub_for_load_visualizations()
    sky._load_visualiations(
        [],
        [],
        [],
        [],
        [],
        sh_coeffs,
    )
    assert sky._sh_glyph_visualizations == []


def test_load_visualizations_sh_coeffs_happy_path():
    """``_load_visualiations`` builds one SH glyph from a valid coeffs tuple."""
    l_max = 8
    n_desc = sum(2 * ell + 1 for ell in range(0, l_max + 1, 2))
    coeffs = np.zeros((1, 1, 1, n_desc), dtype=np.float32)
    coeffs[0, 0, 0, 0] = 1.0
    affine = np.eye(4, dtype=np.float64)
    sh_input = (coeffs, affine, "unit_odf")

    sky = _skyline_stub_for_load_visualizations()
    sky._load_visualiations([], [], [], [], [], [sh_input])

    assert len(sky._sh_glyph_visualizations) == 1
    glyph = sky._sh_glyph_visualizations[0]
    assert isinstance(glyph, SHGlyph3D)
    assert glyph.path == "unit_odf"
    assert glyph.shape == (1, 1, 1)
    assert glyph.viz_type == "sh_glyph"


def _skyline_stub_for_deferred_behaviour():
    """Build ``Skyline``-like object for deferred scene/sync tests.

    Returns
    -------
    Skyline
        Instance with the attributes needed by deferred operation methods.
    """
    obj = Skyline(visualizer_type="stealth")
    obj._is_drawing_ui = True
    obj._pending_scene_ops = []
    obj._pending_sync_requests = []
    obj._refresh_requested = False
    obj.active_image = None
    obj.request_refresh = lambda: setattr(obj, "_refresh_requested", True)
    obj._synchronize_visualizations_from_source = lambda *args, **kwargs: None
    obj._arrange_image_actors = lambda: None
    return obj


def test_enqueue_scene_op_coalesces_same_bound_method():
    """``enqueue_scene_op`` keeps only the latest call for same bound method."""
    sky = _skyline_stub_for_deferred_behaviour()
    calls = []

    class Recorder:
        def record(self, value):
            calls.append(value)

    recorder = Recorder()
    sky.enqueue_scene_op(recorder.record, 1)
    sky.enqueue_scene_op(recorder.record, 2)
    assert len(sky._pending_scene_ops) == 1

    sky._flush_pending_scene_ops()
    assert calls == [2]
    assert sky._refresh_requested


def test_synchronize_visualizations_queues_snapshot_while_drawing():
    """``_synchronize_visualizations`` queues a copied state during UI draw."""
    sky = _skyline_stub_for_deferred_behaviour()

    class SourceViz:
        _synchronize = True

    source = SourceViz()
    new_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    sky._synchronize_visualizations(source, new_state)

    assert len(sky._pending_sync_requests) == 1
    queued_source, queued_state = sky._pending_sync_requests[0]
    assert queued_source is source
    assert np.array_equal(queued_state, new_state)
    assert queued_state is not new_state
    assert sky._refresh_requested


def test_get_reference_slice_state_none_when_empty():
    """``_get_reference_slice_state`` returns None when no sync-capable views exist."""
    sky = Skyline(visualizer_type="stealth")
    sky.active_image = None
    sky._slice_focus_viz = None
    sky._image_visualizations = []
    sky._peak_visualizations = []
    sky._roi_visualizations = []
    sky._surface_visualizations = []
    sky._tractogram_visualizations = []
    sky._sh_glyph_visualizations = []
    assert sky._get_reference_slice_state() is None


def test_get_reference_slice_state_prefers_active_image():
    """``_get_reference_slice_state`` uses ``active_image.state`` when set."""
    sky = Skyline(visualizer_type="stealth")
    sky._slice_focus_viz = None
    img_first = Image3D.__new__(Image3D)
    img_first.state = np.array([1.0, 1.0, 1.0], dtype=float)
    img_active = Image3D.__new__(Image3D)
    img_active.state = np.array([4.0, 5.0, 6.0], dtype=float)
    sky._image_visualizations = [img_first, img_active]
    sky._peak_visualizations = []
    sky._roi_visualizations = []
    sky._surface_visualizations = []
    sky._tractogram_visualizations = []
    sky._sh_glyph_visualizations = []
    sky.active_image = img_active
    ref = sky._get_reference_slice_state()
    assert np.array_equal(ref, np.array([4.0, 5.0, 6.0]))


def test_get_reference_slice_state_uses_slice_focus_when_no_active_image():
    """``_get_reference_slice_state`` follows ``_slice_focus_viz`` when set."""
    sky = Skyline(visualizer_type="stealth")
    sky.active_image = None
    glyph = SHGlyph3D.__new__(SHGlyph3D)
    glyph.state = np.array([2.0, 3.0, 4.0], dtype=float)
    sky._image_visualizations = []
    sky._peak_visualizations = []
    sky._roi_visualizations = []
    sky._surface_visualizations = []
    sky._tractogram_visualizations = []
    sky._sh_glyph_visualizations = [glyph]
    sky._slice_focus_viz = glyph
    ref = sky._get_reference_slice_state()
    assert np.array_equal(ref, np.array([2.0, 3.0, 4.0]))


def test_get_reference_slice_state_reversed_fallback():
    """``_get_reference_slice_state`` uses the last sync-capable vis in stack order."""
    sky = Skyline.__new__(Skyline)
    sky.active_image = None
    sky._slice_focus_viz = None
    g_a = SHGlyph3D.__new__(SHGlyph3D)
    g_a.state = np.array([0.0, 0.0, 0.0], dtype=float)
    g_b = SHGlyph3D.__new__(SHGlyph3D)
    g_b.state = np.array([7.0, 8.0, 9.0], dtype=float)
    sky._image_visualizations = []
    sky._peak_visualizations = []
    sky._roi_visualizations = []
    sky._surface_visualizations = []
    sky._tractogram_visualizations = []
    sky._sh_glyph_visualizations = [g_a, g_b]
    ref = sky._get_reference_slice_state()
    assert np.array_equal(ref, np.array([7.0, 8.0, 9.0]))


def test_get_reference_slice_state_clears_stale_focus():
    """``_get_reference_slice_state`` drops stale ``_slice_focus_viz`` references."""
    sky = Skyline.__new__(Skyline)
    sky.active_image = None
    orphan = SHGlyph3D.__new__(SHGlyph3D)
    sky._slice_focus_viz = orphan
    sky._image_visualizations = []
    sky._peak_visualizations = []
    sky._roi_visualizations = []
    sky._surface_visualizations = []
    sky._tractogram_visualizations = []
    sky._sh_glyph_visualizations = []
    assert sky._get_reference_slice_state() is None
    assert sky._slice_focus_viz is None


def test_apply_reference_slice_state_to_new_visualizations():
    """``_apply_reference_slice_state_to_new_visualizations`` on update."""
    sky = Skyline(visualizer_type="stealth")

    class Viz:
        def __init__(self):
            self.last = None

        def update_state(self, s):
            self.last = np.asarray(s, dtype=float).copy()

    old = Viz()
    new = Viz()
    sky._image_visualizations = [old, new]
    sky._peak_visualizations = []
    sky._sh_glyph_visualizations = []
    ref = np.array([10.0, 20.0, 30.0], dtype=float)
    sky._apply_reference_slice_state_to_new_visualizations(ref, 1, 0, 0)
    assert old.last is None
    assert np.array_equal(new.last, ref)


def test_apply_reference_slice_state_to_new_visualizations_noop_when_none():
    """``_apply_reference_slice_state_to_new_visualizations`` skips when ref is None."""
    sky = Skyline(visualizer_type="stealth")

    class Viz:
        def __init__(self):
            self.called = False

        def update_state(self, s):
            self.called = True

    new = Viz()
    sky._image_visualizations = [new]
    sky._peak_visualizations = []
    sky._sh_glyph_visualizations = []
    sky._apply_reference_slice_state_to_new_visualizations(None, 0, 0, 0)
    assert new.called is False


def test_remove_visualization_clears_slice_focus():
    """``_remove_visualization`` clears ``_slice_focus_viz`` on removal."""
    sky = Skyline(visualizer_type="stealth")
    sky.active_image = None
    sky.UI_window = None
    sky._image_visualizations = []
    sky._peak_visualizations = []
    sky._roi_visualizations = []
    sky._surface_visualizations = []
    sky._tractogram_visualizations = []
    g_keep = SHGlyph3D.__new__(SHGlyph3D)
    g_focus = SHGlyph3D.__new__(SHGlyph3D)
    sky._sh_glyph_visualizations = [g_keep, g_focus]
    sky._slice_focus_viz = g_focus
    sky._remove_visualization(g_focus)
    assert sky._slice_focus_viz is None
    assert sky._sh_glyph_visualizations == [g_keep]


def test_synchronize_visualizations_sets_slice_focus():
    """``_synchronize_visualizations`` stores the source for load-time reference."""
    sky = Skyline(visualizer_type="stealth")
    sky.active_image = None
    sky._is_drawing_ui = False
    sky._pending_sync_requests = []
    sky._slice_focus_viz = None
    sky._image_visualizations = []
    sky._peak_visualizations = []
    sky._roi_visualizations = []
    sky._surface_visualizations = []
    sky._tractogram_visualizations = []
    sky._sh_glyph_visualizations = []
    sky._synchronize_visualizations_from_source = lambda *a, **k: None
    sky._arrange_image_actors = lambda: None
    img = Image3D.__new__(Image3D)
    img._synchronize = True
    sky._synchronize_visualizations(img, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert sky._slice_focus_viz is img
