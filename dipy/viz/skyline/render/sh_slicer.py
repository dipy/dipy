"""SH Glyph Slicer for Skyline."""

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import (
    create_numeric_input,
    render_group,
    thin_slider,
    toggle_button,
)
from dipy.viz.skyline.render.renderer import Visualization
from dipy.viz.skyline.render.sh_billboard import sph_glyph_billboard_sliced

fury_trip_msg = (
    "Skyline requires Fury version 2.0.0a6 or higher."
    " Please upgrade Fury by `pip install -U fury --pre` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.0.0a6",
    trip_msg=fury_trip_msg,
)
if has_fury_v2:
    from fury import apply_transformation
    from fury.actor import Group

imgui_bundle, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")
if has_imgui:
    imgui = imgui_bundle.imgui


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
        input_basis_type = basis_type
    else:
        coeffs, affine, filename, input_basis_type = input

    return SHGlyph3D(
        filename,
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


def _descoteaux_to_fury_standard(coeffs_4d, sh_order, is_left_handed=False):
    """Convert even-order descoteaux07 SH coefficients to Fury's standard basis.

    The legacy descoteaux07 basis uses Im(Y) for m>0 and Re(Y) for m<0, while
    FURY uses cos(mφ) for m>0 and sin(|m|φ) for m<0. Coefficients satisfy
    ``c_fury(l, m) = c_desc(l, -m)``. Left-handed affines additionally flip
    the sign for orders with ``m > 0``.

    Parameters
    ----------
    coeffs_4d : ndarray
        Volume storing descoteaux07 coefficients along the last axis.
    sh_order : int
        Maximum even spherical harmonic order present in the volume.
    is_left_handed : bool, optional
        If True, apply reflection correction for LAS-like orientations.

    Returns
    -------
    ndarray
        Array with the same leading shape as ``coeffs_4d`` and
        ``(sh_order + 1) ** 2`` standard-basis coefficients on the last axis.
    """
    n_std = (sh_order + 1) ** 2
    out = np.zeros(coeffs_4d.shape[:-1] + (n_std,), dtype=coeffs_4d.dtype)

    desc_idx = 0
    for l_val in range(0, sh_order + 1, 2):
        for m in range(-l_val, l_val + 1):
            fury_m = -m
            fury_idx = l_val * l_val + l_val + fury_m

            val = coeffs_4d[..., desc_idx]

            if is_left_handed and m > 0:
                val = -val

            out[..., fury_idx] = val
            desc_idx += 1

    return out


class SHSlicer:
    """Represent ``SHSlicer`` in Skyline.

    Parameters
    ----------
    coeffs_4d : ndarray
        Value for ``coeffs 4d``.
    voxel_sizes : tuple(float, float, float), optional
        Value for ``voxel sizes``.
    scale : float, optional
        Value for ``scale``.
    l_max : int, optional
        Value for ``l max``.
    lut_res : int, optional
        Value for ``lut res``.
    use_hermite : bool, optional
        Value for ``use hermite``.
    mapping_mode : str, optional
        Value for ``mapping mode``.
    mask : ndarray, optional
        Value for ``mask``.
    basis_type : str, optional
        Value for ``basis type``.
    color_type : str, optional
        Value for ``color type``.
    is_left_handed : bool, optional
        Value for ``is left handed``.
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
        is_left_handed=False,
    ):
        """Represent ``SHSlicer`` in Skyline.

        Parameters
        ----------
        coeffs_4d : ndarray
            Value for ``coeffs 4d``.
        voxel_sizes : tuple(float, float, float), optional
            Value for ``voxel sizes``.
        scale : float, optional
            Value for ``scale``.
        l_max : int, optional
            Value for ``l max``.
        lut_res : int, optional
            Value for ``lut res``.
        use_hermite : bool, optional
            Value for ``use hermite``.
        mapping_mode : str, optional
            Value for ``mapping mode``.
        mask : ndarray, optional
            Value for ``mask``.
        basis_type : str, optional
            Value for ``basis type``.
        color_type : str, optional
            Value for ``color type``.
        is_left_handed : bool, optional
            Value for ``is left handed``.
        """
        if basis_type in ("descoteaux", "descoteaux07"):
            coeffs_4d = _descoteaux_to_fury_standard(coeffs_4d, l_max, is_left_handed)
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
        """Handle build for ``SHSlicer``."""
        self._glyph_actor = self._build_volume_actor()
        if self._glyph_actor is not None:
            self.actor.add(self._glyph_actor)
        return self.actor

    def _build_volume_actor(self):
        """Handle build volume actor for ``SHSlicer``."""
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
    """Represent ``SHGlyph3D`` in Skyline.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    coeffs : ndarray
        Value for ``coeffs``.
    affine : ndarray, optional
        Voxel-to-world affine used to position slices in world coordinates.
    render_callback : callable, optional
        Callback used to request a render/update.
    scale : float, optional
        Value for ``scale``.
    l_max : int, optional
        Value for ``l max``.
    lut_res : int, optional
        Value for ``lut res``.
    use_hermite : bool, optional
        Value for ``use hermite``.
    mapping_mode : str, optional
        Value for ``mapping mode``.
    basis_type : str, optional
        Value for ``basis type``.
    color_type : str, optional
        Value for ``color type``.
    mask : ndarray, optional
        Value for ``mask``.
    sync_callback : callable, optional
        Callback used to synchronize state across views.
    """

    def __init__(
        self,
        name,
        coeffs,
        *,
        affine=None,
        render_callback=None,
        scale=2.0,
        l_max=8,
        lut_res=8,
        use_hermite=True,
        mapping_mode="cube",
        basis_type="standard",
        color_type="orientation",
        mask=None,
        sync_callback=None,
    ):
        """Represent ``SHGlyph3D`` in Skyline.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        coeffs : ndarray
            Value for ``coeffs``.
        affine : ndarray, optional
            Voxel-to-world affine used to position slices in world coordinates.
        render_callback : callable, optional
            Callback used to request a render/update.
        scale : float, optional
            Value for ``scale``.
        l_max : int, optional
            Value for ``l max``.
        lut_res : int, optional
            Value for ``lut res``.
        use_hermite : bool, optional
            Value for ``use hermite``.
        mapping_mode : str, optional
            Value for ``mapping mode``.
        basis_type : str, optional
            Value for ``basis type``.
        color_type : str, optional
            Value for ``color type``.
        mask : ndarray, optional
            Value for ``mask``.
        sync_callback : callable, optional
            Callback used to synchronize state across views.
        """
        self.affine = affine
        default_scale = abs(self.affine[0, 0]) if self.affine is not None else scale
        self._voxel_sizes = np.array([1.0, 1.0, 1.0])
        is_left_handed = self.affine is not None and self.affine[0, 0] < 0

        self.shape = coeffs.shape[:3]

        self._slicer = SHSlicer(
            coeffs,
            voxel_sizes=self._voxel_sizes,
            scale=default_scale,
            l_max=l_max,
            lut_res=lut_res,
            use_hermite=use_hermite,
            mapping_mode=mapping_mode,
            mask=mask,
            basis_type=basis_type,
            color_type=color_type,
            is_left_handed=is_left_handed,
        )
        self._slicer.build()
        if affine is not None:
            self._slicer.actor.transform(self.affine)

        super().__init__(name, render_callback)
        self._scale = float(default_scale)
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
            self.state = np.asarray(self.bounds).mean(axis=0).astype(int)
        else:
            self.bounds = np.asarray([lower_bounds, upper_bounds])
            self.state = [self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2]
        self.set_slices()

    @property
    def actor(self):
        """Handle actor for ``SHGlyph3D``.

        Returns
        -------
        Group
            The actor of the SHGlyph3D visualization.
        """
        return self._slicer.actor

    @property
    def voxel_state(self):
        """Handle voxel state for ``SHGlyph3D``.

        Returns
        -------
        np.ndarray
            The voxel state of the SHGlyph3D visualization.
        """
        if self.affine is None:
            return self.state
        voxel_state = apply_transformation(
            np.array([self.state], dtype=np.float32), np.linalg.inv(self.affine)
        )[0]
        return np.round(voxel_state).astype(int)

    def _populate_info(self):
        """Handle  populate info for ``SHGlyph3D``.

        Returns
        -------
        str
            The information of the SHGlyph3D visualization.
        """
        info = f"Dimensions: {self.shape}"
        info += f"\nSH Coefficients: {self._slicer.n_coeffs}"
        info += f"\nSH Order: {self._slicer.l_max}"
        if self.affine is not None:
            info += f"\nVoxel Sizes: {self._voxel_sizes}"
        return info

    def set_slices(self):
        """Handle set slices for ``SHGlyph3D``."""
        for i, axis in enumerate(("x", "y", "z")):
            self._slicer.set_slice(axis, self.voxel_state[i])
            self._last_voxel[i] = self.voxel_state[i]

    def update_state(self, new_state):
        """Handle update state for ``SHGlyph3D``.

        Parameters
        ----------
        new_state : array-like
            New synchronized state for this visualization.
        """
        if self._synchronize:
            self.state = new_state[:3]
            self.apply_scene_op(self.set_slices)

    def set_slice_visibility(self):
        """Handle set slice visibility for ``SHGlyph3D``."""
        for i, axis in enumerate(("x", "y", "z")):
            if self._slice_visibility[i]:
                self._slicer.show_axis(axis)
                self._last_voxel[i] = self.voxel_state[i]
            else:
                self._slicer.hide_axis(axis)
                self._last_voxel[i] = -1

    def render_widgets(self):
        """Handle render widgets for ``SHGlyph3D``."""
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
                self.apply_scene_op(self._slicer.set_scale, new_scale)
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
            self.apply_scene_op(self._slicer.set_opacity, self._opacity / 100.0)

        imgui.spacing()

        axis_labels = ("X", "Y", "Z")

        def _axis_slider_bounds(axis):
            lower = float(min(self.bounds[0][axis], self.bounds[1][axis]))
            upper = float(max(self.bounds[0][axis], self.bounds[1][axis]))
            min_bound = int(np.ceil(lower))
            max_bound = int(np.floor(upper))
            if min_bound > max_bound:
                mid = int(round((lower + upper) * 0.5))
                return mid, mid
            return min_bound, max_bound

        slider_bounds = tuple(_axis_slider_bounds(axis) for axis in range(3))
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
                self.state[idx] = int(round(new))
                self.apply_scene_op(self.set_slices)
                if self._synchronize and self._sync_callback is not None:
                    self._sync_callback(self, self.state)
            self._slice_visibility[idx] = toggle
            self.apply_scene_op(self.set_slice_visibility)
            self._last_voxel[idx] = -1

        imgui.spacing()
