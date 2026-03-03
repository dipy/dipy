"""
dipy/viz/horizon/tab/odf.py
---------------------------
ODF (Orientation Distribution Function) visualization tab for DIPY Horizon.

Adds a full interactive panel for CSD/SHORE/MSMT fODF and Q-ball ODF glyphs
using FURY's odf_slicer actor. Supports per-axis slicing, scale controls,
colormap selection, and SH order clipping.

References
----------
- DIPY Issue #3502: https://github.com/dipy/dipy/issues/3502
- FURY Issue #942:  https://github.com/fury-gl/fury/issues/942
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import warnings

import numpy as np

# FURY imports — guard for environments without display (CI, headless tests)
try:
    from fury import ui

    _FURY_AVAILABLE = True
except ImportError:
    _FURY_AVAILABLE = False
    warnings.warn(
        "FURY not found. ODF visualization in Horizon is disabled. "
        "Install with: pip install fury",
        ImportWarning,
        stacklevel=2,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _validate_sh_coeffs(sh_coeffs: np.ndarray) -> Tuple[int, int]:
    """Return (sh_order, n_coeffs) for a valid SH coefficient array.

    Parameters
    ----------
    sh_coeffs : ndarray, shape (..., n_coeffs)
        Spherical harmonic coefficients. Last axis must match a valid SH order.

    Returns
    -------
    sh_order : int
    n_coeffs : int

    Raises
    ------
    ValueError
        If the last dimension does not correspond to a valid even SH order.
    """
    from dipy.reconst.shm import order_from_ncoef

    n_coeffs = sh_coeffs.shape[-1]
    try:
        sh_order = order_from_ncoef(n_coeffs, full_basis=False)
    except ValueError:
        # Try full (asymmetric) basis
        sh_order = order_from_ncoef(n_coeffs, full_basis=True)
    return sh_order, n_coeffs


def _build_sphere(sh_order: int):
    """Return a DIPY sphere appropriate for the given SH order."""
    from dipy.data import get_sphere

    # Higher SH order → denser sphere for faithful glyph rendering
    if sh_order <= 4:
        return get_sphere("repulsion100")
    elif sh_order <= 8:
        return get_sphere("repulsion200")
    else:
        return get_sphere("repulsion724")


# ---------------------------------------------------------------------------
# Main tab class
# ---------------------------------------------------------------------------


class ODFTab:
    """Interactive Horizon tab for ODF / fODF glyph visualization.

    Parameters
    ----------
    sh_coeffs : ndarray, shape (X, Y, Z, n_coeffs)
        Spherical harmonic coefficients (real, symmetric/asymmetric basis).
    affine : ndarray, shape (4, 4), optional
        Voxel-to-world affine matrix. Defaults to identity.
    mask : ndarray, shape (X, Y, Z), optional
        Boolean mask. Only masked voxels will render ODF glyphs.
    colormap : str, optional
        FURY/matplotlib colormap name.  Default ``"plasma"``.
    scale : float, optional
        Global glyph scale factor.  Default ``2.0``.
    norm : bool, optional
        Normalize each ODF to unit maximum before rendering.  Default ``True``.
    radial_scale : bool, optional
        If True, glyph radius encodes ODF amplitude. Default ``True``.
    sh_basis : str, optional
        ``"tournier07"`` (DIPY default) or ``"descoteaux07"``.
    scene_manager : object, optional
        Reference to Horizon's SceneManager for registering actors.
    """

    name: str = "ODF"
    icon: str = "🔮"  # displayed in tab strip

    def __init__(
        self,
        sh_coeffs: np.ndarray,
        affine: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        colormap: str = "plasma",
        scale: float = 2.0,
        norm: bool = True,
        radial_scale: bool = True,
        sh_basis: str = "tournier07",
        scene_manager=None,
    ):
        if not _FURY_AVAILABLE:
            raise RuntimeError("FURY must be installed to use ODFTab.")

        self._sh_coeffs = sh_coeffs.astype(np.float32)
        self.affine = affine if affine is not None else np.eye(4)
        self.mask = mask
        self.colormap = colormap
        self.scale = scale
        self.norm = norm
        self.radial_scale = radial_scale
        self.sh_basis = sh_basis
        self.scene_manager = scene_manager

        # Volume dimensions
        self._vol_shape = sh_coeffs.shape[:3]
        self._sh_order, self._n_coeffs = _validate_sh_coeffs(sh_coeffs)
        self._sphere = _build_sphere(self._sh_order)

        # Slice indices (start at mid-volume)
        self._slice = {
            "x": self._vol_shape[0] // 2,
            "y": self._vol_shape[1] // 2,
            "z": self._vol_shape[2] // 2,
        }

        # FURY actor handles (one per axis)
        self._actors: dict = {}
        self._panel: Optional[object] = None

        logger.info(
            "ODFTab initialised: volume=%s  SH order=%d  sphere vertices=%d",
            self._vol_shape,
            self._sh_order,
            len(self._sphere.vertices),
        )

    # ------------------------------------------------------------------
    # Actor construction
    # ------------------------------------------------------------------

    def _coeffs_for_slice(self, axis: str) -> np.ndarray:
        """Extract a single-voxel-thick slab of SH coefficients."""
        idx = self._slice[axis]
        if axis == "x":
            slab = self._sh_coeffs[idx : idx + 1, :, :, :]
        elif axis == "y":
            slab = self._sh_coeffs[:, idx : idx + 1, :, :]
        else:
            slab = self._sh_coeffs[:, :, idx : idx + 1, :]
        return slab

    def _mask_for_slice(self, axis: str) -> Optional[np.ndarray]:
        if self.mask is None:
            return None
        idx = self._slice[axis]
        if axis == "x":
            return self.mask[idx : idx + 1, :, :]
        elif axis == "y":
            return self.mask[:, idx : idx + 1, :]
        else:
            return self.mask[:, :, idx : idx + 1]

    def _make_odf_actor(self, axis: str):
        """Create (or recreate) the FURY odf_slicer actor for one axis."""
        from fury.actor import odf_slicer

        coeffs = self._coeffs_for_slice(axis)
        mask = self._mask_for_slice(axis)

        # Convert SH → ODF amplitudes on the sphere
        try:
            from dipy.reconst.shm import sh_to_sf

            sf = sh_to_sf(
                coeffs,
                self._sphere,
                sh_order_max=self._sh_order,
                basis_type=self.sh_basis,
                legacy=False,
            )
        except Exception as exc:
            logger.error("sh_to_sf failed: %s", exc)
            raise

        if self.norm:
            # Per-voxel normalisation
            sf_max = sf.max(axis=-1, keepdims=True)
            sf_max = np.where(sf_max == 0, 1.0, sf_max)
            sf = sf / sf_max

        # Clamp negative lobes (can appear with noisy data / low SH order)
        sf = np.clip(sf, 0, None)

        act = odf_slicer(
            sf,
            sphere=self._sphere,
            scale=self.scale,
            norm=False,  # already normalised above
            radial_scale=self.radial_scale,
            opacity=1.0,
            colormap=self.colormap,
            mask=mask,
            affine=self.affine,
        )

        # Translate actor to correct slice position
        # FURY odf_slicer places glyphs at (i, j, k) voxel centres by default
        return act

    def build_actors(self):
        """Build all three axis ODF actors. Call once before adding to scene."""
        for ax in ("x", "y", "z"):
            try:
                self._actors[ax] = self._make_odf_actor(ax)
                logger.debug("Built ODF actor for axis=%s", ax)
            except Exception as exc:
                logger.warning("Could not build ODF actor (axis=%s): %s", ax, exc)

    def get_actors(self) -> list:
        """Return list of FURY actors currently built."""
        return list(self._actors.values())

    # ------------------------------------------------------------------
    # UI Panel
    # ------------------------------------------------------------------

    def build_panel(self, position: Tuple[int, int] = (0, 200)) -> object:
        """Construct the FURY UI panel for interactive ODF controls.

        Parameters
        ----------
        position : tuple of int
            (x, y) pixel position of the panel in the render window.

        Returns
        -------
        fury.ui.Panel2D
        """
        panel = ui.Panel2D(
            size=(300, 380),
            position=position,
            color=(0.15, 0.15, 0.15),
            opacity=0.88,
            align="left",
        )

        # --- Title ---
        title = ui.TextBlock2D(
            text="ODF Visualization",
            font_size=16,
            bold=True,
            color=(1, 1, 1),
        )
        panel.add_element(title, (0.05, 0.93))

        # --- Slice sliders ---
        for i, (ax, label) in enumerate(
            [("z", "Axial (Z)"), ("y", "Coronal (Y)"), ("x", "Sagittal (X)")]
        ):
            lbl = ui.TextBlock2D(
                text=f"{label}  [{self._slice[ax]}]",
                font_size=12,
                color=(0.9, 0.9, 0.9),
            )
            slider = ui.LineSlider2D(
                min_value=0,
                max_value=self._vol_shape[{"x": 0, "y": 1, "z": 2}[ax]] - 1,
                initial_value=self._slice[ax],
                length=200,
                line_width=3,
                handle_side=18,
                handle_color=(0.4, 0.8, 1.0),
            )

            _axis_labels = {"x": "Sagittal (X)", "y": "Coronal (Y)", "z": "Axial (Z)"}

            def _make_slice_cb(axis_key, text_block, labels=_axis_labels):
                def cb(slider_obj):
                    val = int(slider_obj.value)
                    self._slice[axis_key] = val
                    text_block.message = f"{labels[axis_key]}  [{val}]"
                    self._refresh_actor(axis_key)

                return cb

                return cb

            slider.on_change = _make_slice_cb(ax, lbl)
            y_pos = 0.80 - i * 0.20
            panel.add_element(lbl, (0.05, y_pos + 0.06))
            panel.add_element(slider, (0.05, y_pos))

        # --- Scale slider ---
        scale_lbl = ui.TextBlock2D(
            text=f"Scale  [{self.scale:.1f}]",
            font_size=12,
            color=(0.9, 0.9, 0.9),
        )
        scale_slider = ui.LineSlider2D(
            min_value=0.1,
            max_value=10.0,
            initial_value=self.scale,
            length=200,
            line_width=3,
            handle_side=18,
            handle_color=(1.0, 0.7, 0.3),
        )

        def _scale_cb(slider):
            self.scale = slider.value
            scale_lbl.message = f"Scale  [{self.scale:.1f}]"
            self._refresh_all_actors()

        scale_slider.on_change = _scale_cb
        panel.add_element(scale_lbl, (0.05, 0.22))
        panel.add_element(scale_slider, (0.05, 0.16))

        # --- Normalise checkbox ---
        norm_cb = ui.Checkbox(
            labels=["Normalise ODFs"],
            checked_labels=["Normalise ODFs"] if self.norm else [],
            font_size=12,
            font_color=(0.9, 0.9, 0.9),
        )

        def _norm_toggle(checkbox):
            self.norm = "Normalise ODFs" in checkbox.checked_labels
            self._refresh_all_actors()

        norm_cb.on_change = _norm_toggle
        panel.add_element(norm_cb, (0.05, 0.06))

        self._panel = panel
        return panel

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def _refresh_actor(self, axis: str):
        """Rebuild the actor for a single axis and notify scene manager."""
        old = self._actors.get(axis)
        try:
            new = self._make_odf_actor(axis)
            self._actors[axis] = new
            if self.scene_manager is not None:
                self.scene_manager.replace_actor(old, new)
        except Exception as exc:
            logger.error("Failed to refresh ODF actor (axis=%s): %s", axis, exc)

    def _refresh_all_actors(self):
        for ax in ("x", "y", "z"):
            self._refresh_actor(ax)

    # ------------------------------------------------------------------
    # Serialisation (for session save/restore)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a JSON-serialisable dict of the current tab state."""
        return {
            "slice": dict(self._slice),
            "scale": float(self.scale),
            "norm": bool(self.norm),
            "radial_scale": bool(self.radial_scale),
            "colormap": self.colormap,
            "sh_basis": self.sh_basis,
            "sh_order": int(self._sh_order),
        }

    def set_state(self, state: dict):
        """Restore tab state from a dict (e.g., loaded from JSON)."""
        self._slice.update(state.get("slice", {}))
        self.scale = state.get("scale", self.scale)
        self.norm = state.get("norm", self.norm)
        self.radial_scale = state.get("radial_scale", self.radial_scale)
        self.colormap = state.get("colormap", self.colormap)
        self._refresh_all_actors()
