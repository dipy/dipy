"""
dipy/viz/horizon/tab/tensor.py
------------------------------
Diffusion Tensor Imaging (DTI) visualization tab for DIPY Horizon.

Renders ellipsoidal tensor glyphs via FURY's tensor_slicer actor.
Supports interactive slicing, eigenvalue clipping, FA/MD colormaps,
and affine-aware placement.

References
----------
- DIPY Issue #3502
- FURY tensor_slicer: https://fury.gl/latest/reference/fury.actor.html
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import warnings

import numpy as np

try:
    from fury import actor, ui
    from fury.colormap import create_colormap

    _FURY_AVAILABLE = True
except ImportError:
    _FURY_AVAILABLE = False
    warnings.warn(
        "FURY not found. Tensor visualization disabled.",
        ImportWarning,
        stacklevel=2,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor-derived scalar maps
# ---------------------------------------------------------------------------


def _compute_fa(evals: np.ndarray) -> np.ndarray:
    """Fractional Anisotropy from eigenvalues (shape ..., 3)."""
    l1, l2, l3 = evals[..., 0], evals[..., 1], evals[..., 2]
    mean_d = (l1 + l2 + l3) / 3.0
    num = np.sqrt(0.5 * ((l1 - mean_d) ** 2 + (l2 - mean_d) ** 2 + (l3 - mean_d) ** 2))
    denom = np.sqrt(l1**2 + l2**2 + l3**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.where(denom > 0, num / denom, 0.0)
    return np.clip(fa, 0, 1)


def _compute_md(evals: np.ndarray) -> np.ndarray:
    """Mean Diffusivity."""
    return evals.mean(axis=-1)


def _rgb_from_evecs(evecs: np.ndarray, fa: np.ndarray) -> np.ndarray:
    """FA-weighted primary eigenvector coloring (standard DTI-RGB)."""
    pev = np.abs(evecs[..., :, 0])  # primary eigenvector, shape (X,Y,Z,3)
    rgb = pev * fa[..., np.newaxis]
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Main tab class
# ---------------------------------------------------------------------------


class TensorTab:
    """Interactive Horizon tab for DTI tensor ellipsoid visualization.

    Parameters
    ----------
    evals : ndarray, shape (X, Y, Z, 3)
        Diffusion tensor eigenvalues (λ1 ≥ λ2 ≥ λ3). Units: mm²/s or
        normalised — whichever was produced by TensorModel.fit().
    evecs : ndarray, shape (X, Y, Z, 3, 3)
        Eigenvectors (columns are the three eigenvectors per voxel).
    affine : ndarray, shape (4, 4), optional
        Voxel-to-world affine.  Defaults to identity.
    mask : ndarray, shape (X, Y, Z), optional
        Boolean mask selecting voxels to render.
    colormap : str, optional
        One of ``"fa_rgb"`` (default FA-weighted RGB), ``"fa"``, ``"md"``,
        or any matplotlib colormap name accepted by FURY.
    scale : float, optional
        Ellipsoid scaling factor. Default ``1.0``.
    norm : bool, optional
        Normalise ellipsoid radii to max eigenvalue per voxel.  Default True.
    scene_manager : object, optional
        Reference to Horizon's SceneManager.
    """

    name: str = "Tensor"
    icon: str = "🥚"

    def __init__(
        self,
        evals: np.ndarray,
        evecs: np.ndarray,
        affine: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        colormap: str = "fa_rgb",
        scale: float = 1.0,
        norm: bool = True,
        scene_manager=None,
    ):
        if not _FURY_AVAILABLE:
            raise RuntimeError("FURY must be installed to use TensorTab.")

        if evals.shape[-1] != 3:
            raise ValueError(f"evals last dim must be 3, got {evals.shape[-1]}")
        if evecs.shape[-2:] != (3, 3):
            raise ValueError(
                f"evecs last two dims must be (3,3), got {evecs.shape[-2:]}"
            )
        if evals.shape[:3] != evecs.shape[:3]:
            raise ValueError("evals and evecs spatial dims must match.")

        self.evals = evals.astype(np.float32)
        self.evecs = evecs.astype(np.float32)
        self.affine = affine if affine is not None else np.eye(4)
        self.mask = mask
        self.colormap = colormap
        self.scale = scale
        self.norm = norm
        self.scene_manager = scene_manager

        self._vol_shape = evals.shape[:3]
        self._slice = {
            "x": self._vol_shape[0] // 2,
            "y": self._vol_shape[1] // 2,
            "z": self._vol_shape[2] // 2,
        }
        self._actors: dict = {}
        self._panel: Optional[object] = None

        # Pre-compute scalar maps (cheap, do once)
        self._fa = _compute_fa(self.evals)
        self._md = _compute_md(self.evals)
        self._rgb = _rgb_from_evecs(self.evecs, self._fa)

        logger.info(
            "TensorTab initialised: volume=%s  FA range=[%.3f, %.3f]",
            self._vol_shape,
            float(self._fa.min()),
            float(self._fa.max()),
        )

    # ------------------------------------------------------------------
    # Actor helpers
    # ------------------------------------------------------------------

    def _slab(self, arr: np.ndarray, axis: str) -> np.ndarray:
        idx = self._slice[axis]
        if axis == "x":
            return arr[idx : idx + 1, ...]
        elif axis == "y":
            return arr[:, idx : idx + 1, ...]
        return arr[:, :, idx : idx + 1, ...]

    def _make_tensor_actor(self, axis: str):
        """Build FURY tensor_slicer actor for the given axis slice."""
        ev = self._slab(self.evals, axis)
        evs = self._slab(self.evecs, axis)
        mask = self._slab(self.mask, axis) if self.mask is not None else None

        # Choose coloring strategy
        if self.colormap == "fa_rgb":
            colors = self._slab(self._rgb, axis)
        elif self.colormap == "fa":
            fa_slab = self._slab(self._fa, axis)
            colors = create_colormap(v_value=fa_slab.ravel(), name="hot")
            colors = colors.reshape(*fa_slab.shape, 3)
        elif self.colormap == "md":
            md_slab = self._slab(self._md, axis)
            md_norm = (md_slab - md_slab.min()) / (md_slab.max() - md_slab.min() + 1e-9)
            colors = create_colormap(v_value=md_norm.ravel(), name="viridis")
            colors = colors.reshape(*md_slab.shape, 3)
        else:
            # Generic fallback
            fa_slab = self._slab(self._fa, axis)
            colors = create_colormap(v_value=fa_slab.ravel(), name=self.colormap)
            colors = colors.reshape(*fa_slab.shape, 3)

        # Clip very small eigenvalues → numerical stability
        ev_clipped = np.clip(ev, 1e-8, None)

        act = actor.tensor_slicer(
            evals=ev_clipped,
            evecs=evs,
            affine=self.affine,
            mask=mask,
            scalar_colors=colors,
            scale=self.scale,
            norm=self.norm,
        )
        return act

    def build_actors(self):
        for ax in ("x", "y", "z"):
            try:
                self._actors[ax] = self._make_tensor_actor(ax)
            except Exception as exc:
                logger.warning("Could not build Tensor actor (axis=%s): %s", ax, exc)

    def get_actors(self) -> list:
        return list(self._actors.values())

    # ------------------------------------------------------------------
    # UI panel
    # ------------------------------------------------------------------

    def build_panel(self, position: Tuple[int, int] = (0, 200)) -> object:
        panel = ui.Panel2D(
            size=(300, 420),
            position=position,
            color=(0.12, 0.12, 0.12),
            opacity=0.88,
        )

        title = ui.TextBlock2D(
            text="Tensor Visualization", font_size=16, bold=True, color=(1, 1, 1)
        )
        panel.add_element(title, (0.05, 0.94))

        # FA / MD stats readout
        stats = ui.TextBlock2D(
            text=(
                f"FA  mean={self._fa.mean():.3f}  max={self._fa.max():.3f}\n"
                f"MD  mean={self._md.mean():.4f}"
            ),
            font_size=11,
            color=(0.7, 0.9, 0.7),
        )
        panel.add_element(stats, (0.05, 0.86))

        # Slice sliders
        axes_info = [
            ("z", "Axial  (Z)", 0.73),
            ("y", "Coronal (Y)", 0.56),
            ("x", "Sagittal (X)", 0.39),
        ]
        for ax, label, y_base in axes_info:
            dim = {"x": 0, "y": 1, "z": 2}[ax]
            lbl = ui.TextBlock2D(
                text=f"{label}  [{self._slice[ax]}]",
                font_size=12,
                color=(0.9, 0.9, 0.9),
            )
            sld = ui.LineSlider2D(
                min_value=0,
                max_value=self._vol_shape[dim] - 1,
                initial_value=self._slice[ax],
                length=200,
                line_width=3,
                handle_side=18,
                handle_color=(0.3, 1.0, 0.6),
            )

            def _cb(slider, axis_key=ax, text=lbl, lbl_text=label):
                v = int(slider.value)
                self._slice[axis_key] = v
                text.message = f"{lbl_text}  [{v}]"
                self._refresh_actor(axis_key)

            sld.on_change = _cb
            panel.add_element(lbl, (0.05, y_base + 0.055))
            panel.add_element(sld, (0.05, y_base))

        # Scale
        scale_lbl = ui.TextBlock2D(
            text=f"Scale  [{self.scale:.2f}]", font_size=12, color=(0.9, 0.9, 0.9)
        )
        scale_sld = ui.LineSlider2D(
            min_value=0.1,
            max_value=5.0,
            initial_value=self.scale,
            length=200,
            line_width=3,
            handle_side=18,
            handle_color=(1.0, 0.5, 0.2),
        )

        def _scale_cb(slider):
            self.scale = slider.value
            scale_lbl.message = f"Scale  [{self.scale:.2f}]"
            self._refresh_all_actors()

        scale_sld.on_change = _scale_cb
        panel.add_element(scale_lbl, (0.05, 0.25))
        panel.add_element(scale_sld, (0.05, 0.19))

        # Colormap radio
        cmap_lbl = ui.TextBlock2D(
            text="Color Mode", font_size=12, color=(0.9, 0.9, 0.9)
        )
        cmap_radio = ui.RadioButton(
            labels=["fa_rgb", "fa", "md"],
            checked_labels=[
                self.colormap if self.colormap in ("fa_rgb", "fa", "md") else "fa_rgb"
            ],
            font_size=11,
            font_color=(0.85, 0.85, 0.85),
        )

        def _cmap_cb(rb):
            if rb.checked_labels:
                self.colormap = rb.checked_labels[0]
                self._refresh_all_actors()

        cmap_radio.on_change = _cmap_cb
        panel.add_element(cmap_lbl, (0.05, 0.11))
        panel.add_element(cmap_radio, (0.05, 0.03))

        self._panel = panel
        return panel

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_actor(self, axis: str):
        old = self._actors.get(axis)
        try:
            new = self._make_tensor_actor(axis)
            self._actors[axis] = new
            if self.scene_manager is not None:
                self.scene_manager.replace_actor(old, new)
        except Exception as exc:
            logger.error("Tensor actor refresh failed (axis=%s): %s", axis, exc)

    def _refresh_all_actors(self):
        for ax in ("x", "y", "z"):
            self._refresh_actor(ax)

    def get_state(self) -> dict:
        return {
            "slice": dict(self._slice),
            "scale": float(self.scale),
            "colormap": self.colormap,
            "norm": bool(self.norm),
        }

    def set_state(self, state: dict):
        self._slice.update(state.get("slice", {}))
        self.scale = state.get("scale", self.scale)
        self.colormap = state.get("colormap", self.colormap)
        self.norm = state.get("norm", self.norm)
        self._refresh_all_actors()
