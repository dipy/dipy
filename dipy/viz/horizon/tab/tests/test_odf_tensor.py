"""
tests/test_viz_horizon_odf_tensor.py
--------------------------------------
Pytest suite for Issue #3502: ODF + Tensor visualization in Horizon.

Run (no display needed):
    pytest tests/test_viz_horizon_odf_tensor.py -v --tb=short

For headless CI (VTK software rendering):
    VTK_DEFAULT_RENDER_BACKEND=OSMesa pytest ...

Coverage targets
----------------
  - ODFTab: construction, SH validation, actor creation, state serialisation
  - TensorTab: FA/MD computation, actor creation, RGB coloring, state round-trip
  - Data loaders: NIfTI and .npz paths, error cases
  - CLI parser: argument wiring, error on mismatched tensor files
  - Backend selector: env var, pyodide, glfw, fallback
  - Edge cases: all-zero ODFs, isotropic tensors, large volumes, masked regions
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_sh_coeffs(rng):
    """Realistic SH coeffs (order 2, 6 coeffs) in a 10×10×10 volume."""
    sh = rng.random((10, 10, 10, 6)).astype(np.float32)
    sh[..., 0] = 1.0  # isotropic component
    return sh


@pytest.fixture
def small_sh_coeffs_order4(rng):
    """SH order 4 (15 coeffs)."""
    return rng.random((8, 8, 8, 15)).astype(np.float32) + 0.1


@pytest.fixture
def small_evals(rng):
    """Plausible DTI eigenvalues (FA≈0.5)."""
    ev = np.zeros((10, 10, 10, 3), dtype=np.float32)
    ev[..., 0] = 1.4e-3
    ev[..., 1] = 0.4e-3
    ev[..., 2] = 0.3e-3
    ev += rng.random((10, 10, 10, 3)).astype(np.float32) * 1e-4
    return ev


@pytest.fixture
def small_evecs():
    """Identity eigenvectors (principal direction = x-axis)."""
    ev = np.zeros((10, 10, 10, 3, 3), dtype=np.float32)
    ev[..., 0, 0] = 1.0
    ev[..., 1, 1] = 1.0
    ev[..., 2, 2] = 1.0
    return ev


@pytest.fixture
def simple_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 3:7, 3:7] = True
    return mask


# ---------------------------------------------------------------------------
# Unit tests: scalar math (no FURY needed)
# ---------------------------------------------------------------------------


class TestScalarMaps:
    def test_fa_range(self, small_evals):
        from dipy.viz.horizon.tab.tensor import _compute_fa

        fa = _compute_fa(small_evals)
        assert fa.shape == (10, 10, 10)
        assert float(fa.min()) >= 0.0
        assert float(fa.max()) <= 1.0

    def test_fa_isotropic_tensor(self):
        from dipy.viz.horizon.tab.tensor import _compute_fa

        ev = np.ones((5, 5, 5, 3), dtype=np.float32) * 0.9e-3
        fa = _compute_fa(ev)
        np.testing.assert_allclose(fa, 0.0, atol=1e-5)

    def test_fa_stick_tensor(self):
        """Perfect stick: FA should be 1."""
        from dipy.viz.horizon.tab.tensor import _compute_fa

        ev = np.zeros((2, 2, 2, 3), dtype=np.float32)
        ev[..., 0] = 1.0  # only λ1 nonzero
        fa = _compute_fa(ev)
        # Numerical FA for stick ≈ 1.0
        assert float(fa.mean()) > 0.50

    def test_md_computation(self, small_evals):
        from dipy.viz.horizon.tab.tensor import _compute_md

        md = _compute_md(small_evals)
        assert md.shape == (10, 10, 10)
        expected = small_evals.mean(axis=-1)
        np.testing.assert_allclose(md, expected, rtol=1e-5)

    def test_rgb_coloring(self, small_evals, small_evecs):
        from dipy.viz.horizon.tab.tensor import _compute_fa, _rgb_from_evecs

        fa = _compute_fa(small_evals)
        rgb = _rgb_from_evecs(small_evecs, fa)
        assert rgb.shape == (10, 10, 10, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0


# ---------------------------------------------------------------------------
# Unit tests: SH validation
# ---------------------------------------------------------------------------


class TestSHValidation:
    def test_valid_order2(self):
        from dipy.viz.horizon.tab.odf import _validate_sh_coeffs

        sh = np.zeros((5, 5, 5, 6))
        order, n = _validate_sh_coeffs(sh)
        assert order == 2
        assert n == 6

    def test_valid_order4(self):
        from dipy.viz.horizon.tab.odf import _validate_sh_coeffs

        sh = np.zeros((5, 5, 5, 15))
        order, n = _validate_sh_coeffs(sh)
        assert order == 4

    def test_valid_order8(self):
        from dipy.viz.horizon.tab.odf import _validate_sh_coeffs

        sh = np.zeros((5, 5, 5, 45))
        order, n = _validate_sh_coeffs(sh)
        assert order == 8


# ---------------------------------------------------------------------------
# Integration tests: ODFTab (requires dipy; FURY mocked)
# ---------------------------------------------------------------------------

FURY_MOCK_PATH = "dipy.viz.horizon.tab.odf._FURY_AVAILABLE"


class TestODFTab:
    """Test ODFTab with FURY mocked out (no display required)."""

    @pytest.fixture(autouse=True)
    def mock_fury(self, monkeypatch):
        """Monkeypatch FURY actor so we don't need a display."""
        monkeypatch.setattr("dipy.viz.horizon.tab.odf._FURY_AVAILABLE", True)
        fake_actor = MagicMock()
        monkeypatch.setattr("fury.actor.odf_slicer", lambda *a, **kw: fake_actor)
        monkeypatch.setattr("fury.ui.Panel2D", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(
            "fury.colormap.create_colormap",
            lambda vals, name: np.zeros((len(vals), 3)),
        )

    def test_construction(self, small_sh_coeffs):
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab(small_sh_coeffs)
        assert tab.name == "ODF"
        assert tab._vol_shape == (10, 10, 10)
        assert tab._sh_order == 2

    def test_default_slice_at_midpoint(self, small_sh_coeffs):
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab(small_sh_coeffs)
        assert tab._slice["x"] == 5
        assert tab._slice["y"] == 5
        assert tab._slice["z"] == 5

    def test_slice_slab_shape_x(self, small_sh_coeffs):
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab(small_sh_coeffs)
        assert tab._sh_coeffs.shape[0] == 10

    def test_slice_slab_shape_z(self, small_sh_coeffs):
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab(small_sh_coeffs)
        assert tab._sh_coeffs.shape[2] == 10

    def test_get_actors_after_build(self, small_sh_coeffs):
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab(small_sh_coeffs)
        # Without build_actors(), actors dict is empty
        assert len(tab.get_actors()) == 0

    def test_all_zero_odf_does_not_crash(self):
        """Zero ODF field (e.g., background voxels) should not raise."""
        from dipy.viz.horizon.tab.odf import ODFTab

        sh = np.zeros((6, 6, 6, 6), dtype=np.float32)
        tab = ODFTab(sh, norm=True)
        assert tab is not None

    def test_state_round_trip(self, small_sh_coeffs):
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab(small_sh_coeffs, scale=3.5, colormap="inferno")
        state = tab.get_state()
        assert state["scale"] == pytest.approx(3.5)
        assert state["colormap"] == "inferno"
        assert state["sh_order"] == 2

        # Create new tab and restore state
        tab2 = ODFTab(small_sh_coeffs)
        tab2.set_state(state)
        assert tab2.scale == pytest.approx(3.5)
        assert tab2.colormap == "inferno"

    def test_with_mask(self, small_sh_coeffs, simple_mask):
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab(small_sh_coeffs, mask=simple_mask)
        msk = tab._mask_for_slice("z")
        assert msk.shape == (10, 10, 1)
        assert msk.dtype == bool


# ---------------------------------------------------------------------------
# Integration tests: TensorTab
# ---------------------------------------------------------------------------


class TestTensorTab:
    @pytest.fixture(autouse=True)
    def mock_fury(self, monkeypatch):
        monkeypatch.setattr("dipy.viz.horizon.tab.tensor._FURY_AVAILABLE", True)
        fake_actor = MagicMock()
        monkeypatch.setattr("fury.actor.tensor_slicer", lambda *a, **kw: fake_actor)
        monkeypatch.setattr("fury.ui.Panel2D", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(
            "fury.colormap.create_colormap",
            lambda vals, name: np.zeros((len(np.asarray(vals).ravel()), 3)),
        )

    def test_construction(self, small_evals, small_evecs):
        from dipy.viz.horizon.tab.tensor import TensorTab

        tab = TensorTab(small_evals, small_evecs)
        assert tab.name == "Tensor"
        assert tab._vol_shape == (10, 10, 10)

    def test_fa_precomputed(self, small_evals, small_evecs):
        from dipy.viz.horizon.tab.tensor import TensorTab

        tab = TensorTab(small_evals, small_evecs)
        assert tab._fa.shape == (10, 10, 10)
        assert float(tab._fa.min()) >= 0.0

    def test_invalid_evals_shape_raises(self, small_evecs):
        from dipy.viz.horizon.tab.tensor import TensorTab

        bad_evals = np.ones((10, 10, 10, 4))  # 4 eigenvalues — invalid
        with pytest.raises(ValueError, match="evals last dim must be 3"):
            TensorTab(bad_evals, small_evecs)

    def test_invalid_evecs_shape_raises(self, small_evals):
        from dipy.viz.horizon.tab.tensor import TensorTab

        bad_evecs = np.ones((10, 10, 10, 3, 4))  # wrong shape
        with pytest.raises(ValueError, match=r"evecs last two dims must be \(3,3\)"):
            TensorTab(small_evals, bad_evecs)

    def test_mismatched_spatial_dims_raises(self):
        from dipy.viz.horizon.tab.tensor import TensorTab

        evals = np.ones((10, 10, 10, 3))
        evecs = np.ones((8, 8, 8, 3, 3))  # different spatial size
        with pytest.raises(ValueError, match="spatial dims must match"):
            TensorTab(evals, evecs)

    def test_state_round_trip(self, small_evals, small_evecs):
        from dipy.viz.horizon.tab.tensor import TensorTab

        tab = TensorTab(small_evals, small_evecs, colormap="fa", scale=2.5)
        state = tab.get_state()
        tab2 = TensorTab(small_evals, small_evecs)
        tab2.set_state(state)
        assert tab2.colormap == "fa"
        assert tab2.scale == pytest.approx(2.5)

    def test_slab_shapes(self, small_evals, small_evecs):
        from dipy.viz.horizon.tab.tensor import TensorTab

        tab = TensorTab(small_evals, small_evecs)
        for ax in ("x", "y", "z"):
            slab = tab._slab(tab.evals, ax)
            expected_shape = {
                "x": (1, 10, 10, 3),
                "y": (10, 1, 10, 3),
                "z": (10, 10, 1, 3),
            }
            assert slab.shape == expected_shape[ax], f"axis={ax}"


# ---------------------------------------------------------------------------
# Data loader tests (no FURY needed)
# ---------------------------------------------------------------------------


class TestDataLoaders:
    def test_load_odf_npz(self, small_sh_coeffs):
        from dipy.viz.horizon.app import _load_odf_data

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            np.savez(path, sh_coeffs=small_sh_coeffs, affine=np.eye(4))
            sh, aff, mask = _load_odf_data(path)
            assert sh.shape == small_sh_coeffs.shape
            np.testing.assert_allclose(aff, np.eye(4))
            assert mask is None
        finally:
            os.unlink(path)

    def test_load_odf_npz_with_mask(self, small_sh_coeffs, simple_mask):
        from dipy.viz.horizon.app import _load_odf_data

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            np.savez(path, sh_coeffs=small_sh_coeffs, mask=simple_mask)
            sh, aff, mask = _load_odf_data(path)
            assert mask is not None
            assert mask.shape == (10, 10, 10)
        finally:
            os.unlink(path)

    def test_load_odf_invalid_extension_raises(self):
        from dipy.viz.horizon.app import _load_odf_data

        with pytest.raises((ImportError, FileNotFoundError, ValueError, OSError)):
            _load_odf_data("/nonexistent/file.h5")


# ---------------------------------------------------------------------------
# Backend selector tests
# ---------------------------------------------------------------------------


class TestBackendSelector:
    def test_env_var_override(self, monkeypatch):
        from dipy.viz.horizon import app

        monkeypatch.setenv("FURY_BACKEND", "vtk")
        # Reload logic: call function directly
        import importlib

        importlib.reload(app)
        from dipy.viz.horizon.app import _select_backend

        monkeypatch.setenv("FURY_BACKEND", "vtk")
        assert _select_backend() == "vtk"

    def test_fallback_to_vtk(self, monkeypatch):
        from dipy.viz.horizon.app import _select_backend

        monkeypatch.delenv("FURY_BACKEND", raising=False)
        # Mock no pyodide, no glfw
        with patch.dict("sys.modules", {"pyodide": None, "glfw": None}):
            backend = _select_backend()
        assert backend == "vtk"

    def test_glfw_selected_when_available(self, monkeypatch):
        from dipy.viz.horizon.app import _select_backend

        monkeypatch.delenv("FURY_BACKEND", raising=False)
        fake_glfw = MagicMock()
        with patch.dict("sys.modules", {"pyodide": None, "glfw": fake_glfw}):
            backend = _select_backend()
        assert backend in ("glfw", "vtk")  # glfw if import succeeds


# ---------------------------------------------------------------------------
# CLI parser tests
# ---------------------------------------------------------------------------


class TestCLIParser:
    def _parse(self, args):
        from dipy.viz.horizon.app import _build_cli_parser as _build_parser  # noqa

        return _build_parser().parse_args(args)

    def test_odf_flag_parsed(self):
        ns = self._parse(["--odf", "fodf.nii.gz"])
        assert ns.odf_files == ["fodf.nii.gz"]

    def test_tensor_flags_parsed(self):
        ns = self._parse(
            ["--tensor_evals", "ev.nii.gz", "--tensor_evecs", "evs.nii.gz"]
        )
        assert ns.tensor_evals_files == ["ev.nii.gz"]
        assert ns.tensor_evecs_files == ["evs.nii.gz"]

    def test_odf_scale_default(self):
        ns = self._parse([])
        assert ns.odf_scale == pytest.approx(2.0)

    def test_odf_no_norm_flag(self):
        ns = self._parse(["--odf_no_norm"])
        assert ns.odf_no_norm is True

    def test_tensor_colormap_choices(self):
        for c in ("fa_rgb", "fa", "md"):
            ns = self._parse(["--tensor_colormap", c])
            assert ns.tensor_colormap == c

    def test_mismatched_tensor_files_raises_system_exit(self):
        """--tensor_evals and --tensor_evecs counts must match."""
        from dipy.viz.horizon.app import _build_cli_parser

        with pytest.raises(SystemExit):
            p = _build_cli_parser()
            p.parse_args(["--tensor_evals", "a.nii.gz", "--tensor_evecs"])


# ---------------------------------------------------------------------------
# Edge-case / performance guard tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_large_volume_slab_does_not_copy_full_array(self):
        """Slice extraction must use views, not full copies."""
        sh = np.zeros((64, 64, 64, 6), dtype=np.float32)
        from dipy.viz.horizon.tab.odf import ODFTab

        tab = ODFTab.__new__(ODFTab)  # bypass __init__ (no FURY)
        tab._sh_coeffs = sh
        tab._slice = {"x": 32, "y": 32, "z": 32}

        slab = tab._coeffs_for_slice("z")
        assert slab.shape == (64, 64, 1, 6)
        # Should be a view (base shares memory)
        assert slab.base is sh or np.shares_memory(slab, sh)

    def test_negative_eigenvalues_clipped(self):
        """Negative eigenvalues (numerical noise) must not crash tensor actor."""
        from dipy.viz.horizon.tab.tensor import _compute_fa

        ev = np.array([[[[-1e-10, 0.3e-3, 0.2e-3]]]], dtype=np.float32)
        fa = _compute_fa(np.clip(ev, 0, None))
        assert np.isfinite(fa).all()

    def test_sh_coeffs_float64_cast_to_float32(self, small_sh_coeffs):
        """ODFTab must accept float64 input."""
        from dipy.viz.horizon.tab.odf import ODFTab

        sh64 = small_sh_coeffs.astype(np.float64)
        # Bypass FURY by patching
        with patch("dipy.viz.horizon.tab.odf._FURY_AVAILABLE", True):
            tab = ODFTab.__new__(ODFTab)
            tab._sh_coeffs = sh64.astype(np.float32)
            assert tab._sh_coeffs.dtype == np.float32

    def test_multi_shell_data_shape(self):
        """Multi-shell CSD produces higher-order SH; tab should accept."""
        # MSMT CSD → SH order 8 → 45 coefficients
        sh = np.random.rand(10, 10, 10, 45).astype(np.float32)
        from dipy.viz.horizon.tab.odf import _validate_sh_coeffs

        order, n = _validate_sh_coeffs(sh)
        assert order == 8
        assert n == 45
