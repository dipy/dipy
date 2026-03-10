"""SH Glyph Slicer for Skyline."""

from pathlib import Path

from fury import apply_transformation
from fury.actor import Group
from imgui_bundle import imgui
import numpy as np

from dipy.viz.skyline.UI.elements import (
    create_numeric_input,
    render_group,
    thin_slider,
    toggle_button,
)
from dipy.viz.skyline.render.renderer import Visualization
from dipy.viz.skyline.render.sh_billboard import sph_glyph_billboard_sliced


def create_shm_visualization(
    input,
    idx,
    *,
    render_callback=None,
    scale=1.3,
    l_max=8,
    lut_res=8,
    use_hermite=True,
    mapping_mode="cube",
    basis_type="descoteaux07",
    color_type="orientation",
    mask=None,
    sync_callback=None,
):
    """Create SH glyph visualization from input.

    Parameters
    ----------
    input : tuple
        Tuple of one of the following forms:
        - (coeffs, affine, filename, basis_type)
        - (coeffs, affine, filename)
        - (coeffs, affine)
    idx : int
        Index used for naming when filename is not provided.
    render_callback : callable, optional
        Callback function to be called after rendering.
    scale : float, optional
        Initial per-glyph scale.
    l_max : int, optional
        Maximum SH order.
    lut_res : int, optional
        LUT resolution.
    use_hermite : bool, optional
        Whether to use Hermite analytic normals.
    mapping_mode : str, optional
        Billboard mapping mode.
    basis_type : str, optional
        SH basis convention. Ignored if provided in ``input`` as 4th element.
    color_type : str, optional
        Colour mapping type.
    mask : ndarray, optional
        Boolean mask of valid voxels.
    sync_callback : callable, optional
        Callback to trigger when synchronization is available.

    Returns
    -------
    SHGlyph3D
        The created SH glyph visualization object.

    Raises
    ------
    ValueError
        If input is not a tuple of length 2, 3, or 4.
    """
    if not isinstance(input, tuple) or len(input) not in (2, 3, 4):
        raise ValueError(
            "Input must be a tuple containing (coeffs, affine, filename, basis_type), "
            "(coeffs, affine, filename), or (coeffs, affine) for SH visualization."
        )

    if len(input) == 2:
        coeffs, affine = input
        filename = f"SH_Glyphs_{idx}"
        input_basis_type = basis_type
    elif len(input) == 3:
        coeffs, affine, filename = input
        filename = Path(filename).name if filename is not None else f"SH_Glyphs_{idx}"
        input_basis_type = basis_type
    else:
        coeffs, affine, filename, input_basis_type = input
        filename = Path(filename).name if filename is not None else f"SH_Glyphs_{idx}"

    return SHGlyph3D(
        f"ODFs ({filename})",
        coeffs,
        affine=affine,
        render_callback=render_callback,
        scale=scale,
        l_max=l_max,
        lut_res=lut_res,
        use_hermite=use_hermite,
        mapping_mode=mapping_mode,
        basis_type=input_basis_type,
        color_type=color_type,
        mask=mask,
        sync_callback=sync_callback,
    )


def _descoteaux_to_fury_standard(coeffs_4d, sh_order):
    """Convert legacy descoteaux07 (even-only) SH coeffs to FURY standard.

    The legacy descoteaux07 basis uses Im(Y) for m>0 and Re(Y) for m<0,
    while FURY's standard basis uses cos(mφ) for m>0 and sin(|m|φ) for m<0.
    Empirically B_desc(l, m) == B_fury(l, -m) for all l, m, so the
    conversion is a simple m-sign swap per degree.

    Conversion: c_fury(l, m) = c_desc(l, -m)
    """
    n_std = (sh_order + 1) ** 2
    out = np.zeros(coeffs_4d.shape[:-1] + (n_std,), dtype=coeffs_4d.dtype)
    desc_idx = 0
    for l_val in range(0, sh_order + 1, 2):
        for m in range(-l_val, l_val + 1):
            fury_m = -m
            fury_idx = l_val * l_val + l_val + fury_m
            out[..., fury_idx] = coeffs_4d[..., desc_idx]
            desc_idx += 1
    return out


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
        if basis_type in ("descoteaux", "descoteaux07"):
            coeffs_4d = _descoteaux_to_fury_standard(coeffs_4d, l_max)
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

    def _build_volume_actor(self):
        """Create one billboard actor for every masked voxel."""
        vs = self.voxel_sizes
        X, Y, Z = self.shape

        flat_coeffs = self.coeffs_4d.reshape(-1, self.n_coeffs)
        valid = np.any(flat_coeffs != 0, axis=1)
        if self.mask is not None:
            valid &= self.mask.ravel()
        if not np.any(valid):
            return None

        ix, iy, iz = np.meshgrid(
            np.arange(X, dtype=np.int32),
            np.arange(Y, dtype=np.int32),
            np.arange(Z, dtype=np.int32),
            indexing="ij",
        )
        voxel_coords = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()])

        centers = voxel_coords.astype(np.float32) * vs[np.newaxis, :]

        coeffs_valid = flat_coeffs[valid]
        centers_valid = centers[valid]
        voxel_valid = voxel_coords[valid]

        glyph = sph_glyph_billboard_sliced(
            coeffs_valid,
            centers_valid,
            voxel_valid,
            scale=self.scale,
            l_max=self.l_max,
            color_type=self.color_type,
            lut_res=self.lut_res,
            use_hermite=self.use_hermite,
            mapping_mode=self.mapping_mode,
        )
        return glyph

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
    sync_callback : callable, optional
        Callback to trigger when slice syncing from linked image reference.
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
        sync_callback=None,
    ):
        self.affine = affine
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
        if affine is not None:
            self._slicer.actor.transform(self.affine)

        super().__init__(name, render_callback)
        self._scale = float(scale)
        self._opacity = 100
        self._slice_visibility = [True, True, True]
        self._synchronize = True
        self._sync_callback = sync_callback

        self._last_voxel = [-1, -1, -1]

        lower_bounds = np.zeros(3)
        upper_bounds = np.array(coeffs.shape[:3]) - 1

        if self.affine is not None:
            self.bounds = apply_transformation(
                np.array([lower_bounds, upper_bounds]), self.affine
            )
            self.state = apply_transformation(
                np.array(
                    [[self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2]]
                ),
                self.affine,
            )[0].astype(int)
        else:
            self.bounds = np.asarray([lower_bounds, upper_bounds])
            self.state = [self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2]
        self.set_slices()

    @property
    def actor(self):
        return self._slicer.actor

    @property
    def voxel_state(self):
        if self.affine is None:
            return self.state
        voxel_state = apply_transformation(
            np.array([self.state], dtype=np.float32), np.linalg.inv(self.affine)
        )[0]
        return np.round(voxel_state).astype(int)

    def _populate_info(self):
        info = f"Dimensions: {self.shape}"
        info += f"\nSH Coefficients: {self._slicer.n_coeffs}"
        info += f"\nSH Order: {self._slicer.l_max}"
        if self.affine is not None:
            info += f"\nVoxel Sizes: {self._voxel_sizes}"
        return info

    def set_slices(self):
        for i, axis in enumerate(("x", "y", "z")):
            self._slicer.set_slice(axis, self.voxel_state[i])
            self._last_voxel[i] = self.voxel_state[i]

    def update_state(self, new_state):
        if self._synchronize:
            self.state = new_state
            self.set_slices()

    def set_slice_visibility(self):
        for i, axis in enumerate(("x", "y", "z")):
            if self._slice_visibility[i]:
                self._slicer.show_axis(axis)
                self._last_voxel[i] = self.voxel_state[i]
            else:
                self._slicer.hide_axis(axis)
                self._last_voxel[i] = -1

    def render_widgets(self):
        changed, new = toggle_button(self._synchronize, label="Synchronize Slices")
        if changed:
            self._synchronize = new

        imgui.spacing()

        changed, new_scale = create_numeric_input(
            "Scale", self._scale, value_type="float", format="%.1f", step=0.1, height=24
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
        slider_bounds = (
            (int(self.bounds[0][0] + 1), int(self.bounds[1][0] - 1)),
            (int(self.bounds[0][1] + 1), int(self.bounds[1][1] - 1)),
            (int(self.bounds[0][2] + 1), int(self.bounds[1][2] - 1)),
        )
        slicers = []
        for axis, label in enumerate(axis_labels):
            min_bound, max_bound = slider_bounds[axis]
            slicers.append(
                (
                    thin_slider,
                    (label, self.state[axis], min_bound, max_bound),
                    {
                        "value_type": "float",
                        "text_format": ".0f",
                        "step": 1,
                        "show_toggle": True,
                        "toggle": self._slice_visibility[axis],
                    },
                )
            )
        render_data = render_group("Slice", slicers)
        for idx, (changed, new, toggle) in enumerate(render_data):
            if changed:
                self.state[idx] = new
                self.set_slices()
                self._synchronize and self._sync_callback(self, self.state)
            self._slice_visibility[idx] = toggle
            self.set_slice_visibility()
            self._last_voxel[idx] = -1

        imgui.spacing()
