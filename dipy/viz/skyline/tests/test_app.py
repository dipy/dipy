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
