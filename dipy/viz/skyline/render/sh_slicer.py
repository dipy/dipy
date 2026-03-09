"""SH Glyph Slicer for Skyline."""

import numpy as np

from fury.actor import Group
from imgui_bundle import icons_fontawesome_6, imgui

from dipy.reconst.shm import convert_sh_to_full_basis
from dipy.viz.sh_billboard import sph_glyph_billboard_sliced
from dipy.viz.skyline.UI.elements import render_group, thin_slider
from dipy.viz.skyline.UI.theme import THEME
from dipy.viz.skyline.render.renderer import Visualization


class SHSlicer:
    """Single-actor SH glyph slicer.

    Every valid voxel is packed into **one** ``SphGlyphBillboard``
    actor with per-glyph ``(ix, iy, iz)`` stored on the GPU.
    Three material uniforms (``active_slice_x/y/z``) select which
    slices are visible — switching is a uniform update, zero rebuild.

    Parameters
    ----------
    coeffs_4d : ndarray, shape (X, Y, Z, n_coeffs)
        SH coefficients.
    voxel_sizes : array-like, shape (3,)
        Physical voxel sizes in mm.
    scale : float
        Per-glyph scale factor.
    l_max : int
        Maximum SH order.
    mask : ndarray, optional
        Boolean mask of valid voxels.
    basis_type : str
        SH basis convention.
    color_type : str
        Colour mapping type.
    """

    def __init__(
        self,
        coeffs_4d,
        voxel_sizes=(1.0, 1.0, 1.0),
        scale=1.0,
        l_max=8,
        lut_res=32,
        use_hermite=True,
        mapping_mode="cube",
        mask=None,
        basis_type="standard",
        color_type="orientation",
    ):
        # Auto-convert descoteaux (even-order-only) basis to standard (full)
        if basis_type in ("descoteaux", "descoteaux07"):
            coeffs_4d = convert_sh_to_full_basis(coeffs_4d)
            basis_type = "standard"

        self.coeffs_4d = coeffs_4d
        self.shape = coeffs_4d.shape[:3]
        self.n_coeffs = coeffs_4d.shape[-1]
        self.voxel_sizes = np.array(voxel_sizes, dtype=float)
        self.scale = scale
        self.l_max = l_max
        self.lut_res = lut_res
        self.use_hermite = use_hermite
        self.mapping_mode = mapping_mode
        self.mask = mask
        self.basis_type = basis_type
        self.color_type = color_type

        self._cur = {"x": -1, "y": -1, "z": -1}
        self._opacity = 1.0
        self.actor = Group()
        self._glyph_actor = None

    def build(self):
        """Build the single actor containing every valid voxel."""
        self._glyph_actor = self._build_volume_actor()
        if self._glyph_actor is not None:
            self.actor.add(self._glyph_actor)
        return self.actor

    # -- builder -----------------------------------------------------------

    def _build_volume_actor(self):
        """Create one billboard actor for every masked voxel."""
        vs = self.voxel_sizes
        X, Y, Z = self.shape

        # Flatten and mask
        flat_coeffs = self.coeffs_4d.reshape(-1, self.n_coeffs)
        valid = np.any(flat_coeffs != 0, axis=1)
        if self.mask is not None:
            valid &= self.mask.ravel()
        if not np.any(valid):
            return None

        # Integer voxel coordinates for every valid voxel
        ix, iy, iz = np.meshgrid(
            np.arange(X, dtype=np.int32),
            np.arange(Y, dtype=np.int32),
            np.arange(Z, dtype=np.int32),
            indexing="ij",
        )
        voxel_coords = np.column_stack([
            ix.ravel(), iy.ravel(), iz.ravel()
        ])  # (X*Y*Z, 3)

        # World-space centres
        centers = voxel_coords.astype(np.float32) * vs[np.newaxis, :]

        coeffs_valid = flat_coeffs[valid]
        centers_valid = centers[valid]
        voxel_valid = voxel_coords[valid]

        print(
            f"  [SHSlicer] {len(coeffs_valid)} valid glyphs "
            f"out of {X * Y * Z} voxels"
        )

        glyph = sph_glyph_billboard_sliced(
            coeffs_valid,
            centers_valid,
            voxel_valid,
            scale=self.scale,
            l_max=self.l_max,
            basis_type=self.basis_type,
            color_type=self.color_type,
            lut_res=self.lut_res,
            use_hermite=self.use_hermite,
            mapping_mode=self.mapping_mode,
        )
        return glyph

    # -- slice visibility --------------------------------------------------

    def set_slice(self, axis, idx):
        """Show slice *idx* on *axis* via uniform update."""
        dim = {"x": 0, "y": 1, "z": 2}[axis]
        idx = int(np.clip(idx, 0, self.shape[dim] - 1))
        if idx == self._cur[axis]:
            return
        if self._glyph_actor is not None:
            attr = f"active_slice_{axis}"
            setattr(self._glyph_actor.material, attr, idx)
        self._cur[axis] = idx

    def hide_axis(self, axis):
        """Hide all slices for *axis*."""
        if self._glyph_actor is not None:
            setattr(self._glyph_actor.material, f"vis_{axis}", 0)
        self._cur[axis] = -1

    def show_axis(self, axis):
        """Enable axis visibility."""
        if self._glyph_actor is not None:
            setattr(self._glyph_actor.material, f"vis_{axis}", 1)

    def set_scale(self, new_scale):
        """Update scale on the actor."""
        ratio = float(new_scale) / float(self.scale) if self.scale > 0 else 1.0
        if abs(ratio - 1.0) < 1e-6:
            return
        self.scale = float(new_scale)
        a = self._glyph_actor
        if a is not None:
            a.material.scale = float(new_scale)
            a.geometry.normals.data[:, :2] *= ratio
            a.geometry.normals.update_full()

    def set_opacity(self, opacity):
        """Set opacity."""
        self._opacity = float(opacity)
        a = self._glyph_actor
        if a is not None:
            a.material.opacity = float(opacity)
            a.material.alpha_mode = "blend" if opacity < 1.0 else "solid"


class SHGlyph3D(Visualization):
    """SH glyph slicer visualization for Skyline.

    Parameters
    ----------
    name : str
        Display name in the UI sidebar.
    coeffs : ndarray, shape (X, Y, Z, n_coeffs)
        SH coefficients.
    affine : ndarray (4, 4), optional
        Voxel-to-world affine.
    render_callback : callable, optional
        Callback to trigger a scene re-render.
    scale : float
        Initial per-glyph scale.
    l_max : int
        Maximum SH order.
    lut_res : int
        LUT resolution.
    use_hermite : bool
        Whether to use Hermite analytic normals.
    mapping_mode : str
        Billboard mapping mode.
    basis_type : str
        SH basis convention.
    color_type : str
        Colour mapping.
    mask : ndarray, optional
        Boolean mask of valid voxels.
    """

    def __init__(
        self,
        name,
        coeffs,
        *,
        affine=None,
        render_callback=None,
        scale=1.0,
        l_max=8,
        lut_res=8,
        use_hermite=True,
        mapping_mode="cube",
        basis_type="standard",
        color_type="orientation",
        mask=None,
    ):
        self.affine = affine
        if affine is not None:
            self._voxel_sizes = np.abs(np.diag(affine)[:3]).astype(float)
        else:
            self._voxel_sizes = np.array([1.0, 1.0, 1.0])

        self.shape = coeffs.shape[:3]

        self._slicer = SHSlicer(
            coeffs,
            voxel_sizes=self._voxel_sizes,
            scale=scale,
            l_max=l_max,
            lut_res=lut_res,
            use_hermite=use_hermite,
            mapping_mode=mapping_mode,
            mask=mask,
            basis_type=basis_type,
            color_type=color_type,
        )
        self._slicer.build()

        super().__init__(name, render_callback)
        self._scale = float(scale)
        self._opacity = 100
        self._slice_visibility = [True, True, True]

        self._image_ref = None
        self._last_voxel = [-1, -1, -1]

        mid = [s // 2 for s in self.shape]
        for i, axis in enumerate(("x", "y", "z")):
            self._slicer.set_slice(axis, mid[i])
            self._last_voxel[i] = mid[i]

        self.bounds = [
            [0, 0, 0],
            list(np.array(self.shape) * self._voxel_sizes),
        ]

    def set_image_ref(self, image_viz):
        """Link to an :class:`Image3D` for slice syncing."""
        self._image_ref = image_viz

    @property
    def actor(self):
        return self._slicer.actor

    def _populate_info(self):
        info = f"Dimensions: {self.shape}"
        info += f"\nSH Coefficients: {self._slicer.n_coeffs}"
        info += f"\nSH Order: {self._slicer.l_max}"
        if self.affine is not None:
            info += f"\nVoxel Sizes: {self._voxel_sizes}"
        return info

    def _world_to_voxel(self, world_pos):
        vs = self._voxel_sizes
        return [
            int(np.clip(np.round(world_pos[i] / vs[i]), 0, self.shape[i] - 1))
            for i in range(3)
        ]

    def _sync_from_image(self):
        if self._image_ref is None:
            return
        state = self._image_ref.state
        voxel = self._world_to_voxel(state)
        axes = ("x", "y", "z")
        for i, axis in enumerate(axes):
            if self._slice_visibility[i]:
                self._slicer.show_axis(axis)
                if voxel[i] != self._last_voxel[i]:
                    self._slicer.set_slice(axis, voxel[i])
                    self._last_voxel[i] = voxel[i]
            else:
                self._slicer.hide_axis(axis)
                self._last_voxel[i] = -1

    def renderer(self, name, is_open):
        self._sync_from_image()
        return super().renderer(name, is_open)

    def render_widgets(self):
        changed, new_scale = thin_slider(
            "Scale",
            self._scale,
            0.1,
            2.0,
            value_type="float",
            text_format=".2f",
            step=0.1,
        )
        if changed:
            new_scale = float(new_scale)
            if abs(new_scale - self._scale) > 1e-4:
                self._slicer.set_scale(new_scale)
                self._scale = new_scale

        imgui.spacing()

        changed, new_op = thin_slider(
            "Opacity",
            self._opacity,
            0,
            100,
            value_type="int",
            text_format=".0f",
            value_unit="%",
            step=1,
        )
        if changed:
            self._opacity = int(new_op)
            self._slicer.set_opacity(self._opacity / 100.0)

        imgui.spacing()

        axis_labels = ("X", "Y", "Z")
        for i, label in enumerate(axis_labels):
            show_icon = (
                icons_fontawesome_6.ICON_FA_CIRCLE_DOT
                if self._slice_visibility[i]
                else icons_fontawesome_6.ICON_FA_CIRCLE
            )
            color = THEME["primary"] if self._slice_visibility[i] else THEME["text"]
            imgui.text_colored(color, f"{show_icon}  {label}")
            if imgui.is_item_clicked():
                self._slice_visibility[i] = not self._slice_visibility[i]
                self._last_voxel[i] = -1
            if i < len(axis_labels) - 1:
                imgui.same_line(0, 16)

        imgui.spacing()
