"""SH Glyph Slicer for Skyline."""

import numpy as np

from fury.actor import Group
from imgui_bundle import imgui

from dipy.viz.sh_billboard import sph_glyph_billboard
from dipy.viz.skyline.UI.elements import render_group, thin_slider
from dipy.viz.skyline.render.renderer import Visualization


class SHSlicer:
    """Lazy per-slice billboard SH glyph slicer.

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
    lut_res : int
        LUT resolution for billboard precomputation.
    use_hermite : bool
        Use Hermite analytic normals.
    mapping_mode : str
        Billboard mapping mode.
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

        self._slice_actors = {
            "x": [None] * self.shape[0],
            "y": [None] * self.shape[1],
            "z": [None] * self.shape[2],
        }
        self._built = {
            "x": [False] * self.shape[0],
            "y": [False] * self.shape[1],
            "z": [False] * self.shape[2],
        }
        self._cur = {"x": -1, "y": -1, "z": -1}

        self._opacity = 1.0
        self.actor = Group()

    def build(self):
        """Mark the slicer as ready (slices are built lazily)."""
        return self.actor

    # -- lazy builders -----------------------------------------------------

    def _ensure_slice(self, axis, idx):
        if idx < 0 or idx >= self.shape[{"x": 0, "y": 1, "z": 2}[axis]]:
            return
        if self._built[axis][idx]:
            return
        sl = {
            "x": self.coeffs_4d[idx, :, :, :],
            "y": self.coeffs_4d[:, idx, :, :],
            "z": self.coeffs_4d[:, :, idx, :],
        }[axis]
        a = self._build_slice_actor(sl, idx, axis)
        if a is not None:
            self._slice_actors[axis][idx] = a
            a.visible = False
            self.actor.add(a)
        self._built[axis][idx] = True

    def _build_slice_actor(self, slice_coeffs, slice_idx, axis):
        vs = self.voxel_sizes
        axis_dim = {"x": 0, "y": 1, "z": 2}[axis]
        dims = [d for d in range(3) if d != axis_dim]
        n0, n1 = slice_coeffs.shape[0], slice_coeffs.shape[1]
        positions = np.zeros((n0 * n1, 3), dtype=np.float32)
        for i in range(n0):
            for j in range(n1):
                pos = [0.0, 0.0, 0.0]
                pos[axis_dim] = slice_idx * vs[axis_dim]
                pos[dims[0]] = i * vs[dims[0]]
                pos[dims[1]] = j * vs[dims[1]]
                positions[i * n1 + j] = pos

        flat_coeffs = slice_coeffs.reshape(-1, self.n_coeffs)
        valid_mask = np.any(flat_coeffs != 0, axis=1)
        if self.mask is not None:
            slicing = [slice(None)] * 3
            slicing[axis_dim] = slice_idx
            sm = self.mask[tuple(slicing)].ravel()
            valid_mask &= sm
        if not np.any(valid_mask):
            return None

        glyph = sph_glyph_billboard(
            flat_coeffs[valid_mask],
            centers=positions[valid_mask],
            scale=self.scale,
            l_max=self.l_max,
            basis_type=self.basis_type,
            color_type=self.color_type,
            use_precomputation=True,
            use_bicubic=True,
            use_hermite=self.use_hermite,
            mapping_mode=self.mapping_mode,
            lut_res=self.lut_res,
        )
        if self._opacity < 1.0:
            glyph.material.opacity = self._opacity
            glyph.material.alpha_mode = "blend"
        return glyph

    # -- slice visibility --------------------------------------------------

    def set_slice(self, axis, idx):
        """Show exactly one slice on *axis*, hide all others."""
        dim = {"x": 0, "y": 1, "z": 2}[axis]
        idx = int(np.clip(idx, 0, self.shape[dim] - 1))
        if idx == self._cur[axis]:
            return
        self._ensure_slice(axis, idx)
        for i, a in enumerate(self._slice_actors[axis]):
            if a is not None:
                a.visible = (i == idx)
        self._cur[axis] = idx

    def slide_to(self, axis, world_pos):
        """Translate the visible slice actor to a world coordinate."""
        cur = self._cur[axis]
        if cur < 0:
            return
        a = self._slice_actors[axis][cur]
        if a is None:
            return
        dim = {"x": 0, "y": 1, "z": 2}[axis]
        built_pos = cur * self.voxel_sizes[dim]
        pos = list(a.local.position)
        pos[dim] = world_pos - built_pos
        a.local.position = tuple(pos)

    def hide_axis(self, axis):
        """Hide all slices for *axis*."""
        for a in self._slice_actors[axis]:
            if a is not None:
                a.visible = False
        self._cur[axis] = -1

    def set_scale(self, new_scale):
        """Update scale on all existing actors."""
        ratio = float(new_scale) / float(self.scale) if self.scale > 0 else 1.0
        if abs(ratio - 1.0) < 1e-6:
            return
        self.scale = float(new_scale)
        for actors in self._slice_actors.values():
            for a in actors:
                if a is not None:
                    a.material.scale = float(new_scale)
                    a.geometry.normals.data[:, :2] *= ratio
                    a.geometry.normals.update_full()

    def set_opacity(self, opacity):
        """Set opacity on every child actor."""
        self._opacity = float(opacity)
        blend = opacity < 1.0
        for actors in self._slice_actors.values():
            for a in actors:
                if a is not None:
                    a.material.opacity = float(opacity)
                    a.material.alpha_mode = "blend" if blend else "opaque"


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
                if voxel[i] != self._last_voxel[i]:
                    self._slicer.set_slice(axis, voxel[i])
                    self._last_voxel[i] = voxel[i]
                self._slicer.slide_to(axis, state[i])
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
        slicers = []
        for i, label in enumerate(axis_labels):
            dim_max = int(self.bounds[1][i] - 1)
            world_val = (
                self._image_ref.state[i] if self._image_ref is not None else 0
            )
            slicers.append(
                (
                    thin_slider,
                    (label, world_val, int(self.bounds[0][i] + 1), dim_max),
                    {
                        "value_type": "float",
                        "text_format": ".0f",
                        "step": 1,
                        "show_toggle": True,
                        "toggle": self._slice_visibility[i],
                    },
                )
            )
        render_data = render_group("Axes", slicers)
        for idx, (changed, _new, toggle) in enumerate(render_data):
            if self._slice_visibility[idx] != toggle:
                self._slice_visibility[idx] = toggle
                self._last_voxel[idx] = -1

        imgui.spacing()
