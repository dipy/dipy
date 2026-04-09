import numpy as np
import pytest

pytest.importorskip("fury")

from dipy.viz.skyline.app import Skyline  # noqa: E402
from dipy.viz.skyline.render.sh_slicer import SHGlyph3D  # noqa: E402


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
    obj = Skyline.__new__(Skyline)
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
