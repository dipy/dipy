"""SH glyph slicer for Skyline -- the entry point for ODF visualization.

Builds a GPU-accelerated 3-D visualization of orientation distribution
functions (ODFs) from a 4-D array of spherical-harmonic coefficients and a
voxel-to-world affine.  ``create_shm_visualization`` unpacks the input tuple
into a :class:`SHGlyph3D`, which owns a :class:`SHSlicer` that builds the
billboard actor via
:func:`~dipy.viz.skyline.render.sh_billboard.sph_glyph_billboard_sliced`
and drives its per-axis slice uniforms.  See individual class/method
docstrings for coordinate-space and slicing details.
"""

import numpy as np

from dipy.reconst.shm import calculate_max_order
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import (
    create_numeric_input,
    render_group,
    thin_slider,
    toggle_button,
)
from dipy.viz.skyline.render.renderer import (
    Visualization,
    affine_voxel_sizes,
    format_affine_info,
    slice_slider_bounds,
    slice_slider_values_from_state,
    slice_state_from_slider_values,
)
from dipy.viz.skyline.render.sh_billboard import sph_glyph_billboard_sliced

fury_trip_msg = (
    "Skyline requires Fury version 2.0.0 or higher."
    " Please upgrade Fury by `pip install -U fury --pre` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.0.0",
    trip_msg=fury_trip_msg,
)
if has_fury_v2:
    from fury import apply_transformation
    from fury.actor import Group

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
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

        - ``(coeffs, affine, filename, basis_type)``
        - ``(coeffs, affine, filename)``
        - ``(coeffs, affine)``

        A ``basis_type`` present as the 4th tuple element overrides the
        ``basis_type`` keyword argument.
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
    basis_type : str, optional
        SH basis convention. Ignored if provided in ``input`` as 4th element.
    color_type : str, optional
        Color mapping type.
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
        basis_type=input_basis_type,
        color_type=color_type,
        mask=mask,
        sync_callback=sync_callback,
    )


def _descoteaux_to_fury_standard(coeffs_4d, sh_order):
    """Convert even-order descoteaux07 SH coefficients to Fury's standard basis.

    The legacy descoteaux07 basis uses Im(Y) for m>0 and Re(Y) for m<0, while
    FURY uses cos(mφ) for m>0 and sin(|m|φ) for m<0. Coefficients satisfy
    ``c_fury(l, m) = c_desc(l, -m)``.

    Parameters
    ----------
    coeffs_4d : ndarray
        Volume storing descoteaux07 coefficients along the last axis.
    sh_order : int
        Maximum even spherical harmonic order present in the volume.

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

            out[..., fury_idx] = coeffs_4d[..., desc_idx]
            desc_idx += 1

    return out


class SHSlicer:
    """Build and drive the single billboard actor backing an ODF visualization.

    Owns the flattened, non-zero-only glyph data (coefficients, model-space
    centers, voxel indices) passed to :func:`sph_glyph_billboard_sliced`,
    and forwards per-axis slice/visibility/scale/opacity changes to that
    actor's material without ever rebuilding the geometry.

    Parameters
    ----------
    coeffs_4d : ndarray, shape (X, Y, Z, C)
        SH coefficients per voxel.  Converted from ``descoteaux07`` to
        Fury's standard basis on construction if needed.
    scale : float, optional
        Uniform billboard size multiplier relative to estimated SH radii.
    l_max : int, optional
        Maximum SH order to shade.  For ``descoteaux``/``descoteaux07``
        input, capped to the order implied by ``coeffs_4d``'s last axis
        when that is lower; for ``standard`` input it must not exceed that
        order (raises ``ValueError`` downstream otherwise).
    lut_res : int, optional
        Cube-map Hermite LUT resolution per face edge.
    mask : ndarray of bool, shape (X, Y, Z), optional
        When given, voxels outside the mask are excluded even if their
        coefficients are non-zero.
    basis_type : {"standard", "descoteaux", "descoteaux07"}, optional
        SH basis convention of ``coeffs_4d``.
    color_type : {"orientation", "sign"}, optional
        Glyph coloring: direction-mapped hue, or a two-color sign split.
    """

    def __init__(
        self,
        coeffs_4d,
        *,
        scale=1.0,
        l_max=8,
        lut_res=32,
        mask=None,
        basis_type="standard",
        color_type="orientation",
    ):
        """Initialize the billboard actor driver.

        Parameters
        ----------
        coeffs_4d : ndarray, shape (X, Y, Z, C)
            SH coefficients per voxel.  Converted from ``descoteaux07`` to
            Fury's standard basis on construction if needed.
        scale : float, optional
            Uniform billboard size multiplier relative to estimated SH radii.
        l_max : int, optional
            Maximum SH order to shade.  For ``descoteaux``/``descoteaux07``
            input, capped to the order implied by ``coeffs_4d``'s last axis
            when that is lower; for ``standard`` input it must not exceed
            that order (raises ``ValueError`` downstream otherwise).
        lut_res : int, optional
            Cube-map Hermite LUT resolution per face edge.
        mask : ndarray of bool, shape (X, Y, Z), optional
            When given, voxels outside the mask are excluded even if their
            coefficients are non-zero.
        basis_type : {"standard", "descoteaux", "descoteaux07"}, optional
            SH basis convention of ``coeffs_4d``.
        color_type : {"orientation", "sign"}, optional
            Glyph coloring: direction-mapped hue, or a two-color sign split.
        """
        if basis_type in ("descoteaux", "descoteaux07"):
            data_sh_order = calculate_max_order(coeffs_4d.shape[-1])
            l_max = min(l_max, data_sh_order)
            coeffs_4d = _descoteaux_to_fury_standard(coeffs_4d, l_max)
            basis_type = "standard"

        self.coeffs_4d = coeffs_4d
        self.shape = coeffs_4d.shape[:3]
        self.n_coeffs = coeffs_4d.shape[-1]
        self.scale = scale
        self.l_max = l_max
        self.lut_res = lut_res
        self.mask = mask
        self.basis_type = basis_type
        self.color_type = color_type

        self._cur = {"x": -1, "y": -1, "z": -1}
        self._opacity = 1.0
        self.actor = Group()
        self._glyph_actor = None

    def build(self):
        """Build the billboard actor and add it to :attr:`actor`.

        Safe to call when every voxel is zero (or masked out): the group
        is then left empty and :attr:`_glyph_actor` stays ``None``.

        Returns
        -------
        Group
            The (possibly empty) parent group holding the billboard actor.
        """
        self._glyph_actor = self._build_volume_actor()
        if self._glyph_actor is not None:
            self.actor.add(self._glyph_actor)
        return self.actor

    def _build_volume_actor(self):
        """Flatten non-zero voxels and build the billboard actor for them.

        Model-space glyph centers are the raw integer voxel indices
        ``(ix, iy, iz)``; the caller (:class:`SHGlyph3D`) applies the full
        voxel-to-world affine once, as a group transform, on top of this.

        Returns
        -------
        SphGlyphBillboard or None
            ``None`` when no voxel has non-zero coefficients (after
            masking), otherwise the actor from
            :func:`sph_glyph_billboard_sliced`.
        """
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

        centers = voxel_coords.astype(np.float32)

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
        )
        return glyph

    def set_slice(self, axis, idx):
        """Move the active slice plane on one axis to a world-space position.

        A no-op when ``idx`` matches the axis's current position, so
        repeated calls from a UI slider don't trigger redundant GPU
        uniform uploads.

        Parameters
        ----------
        axis : {"x", "y", "z"}
            Which per-axis slice-position uniform to update.
        idx : float
            World-space coordinate of the new slice plane along ``axis``.
        """
        if idx == self._cur[axis]:
            return
        if self._glyph_actor is not None:
            attr = f"active_slice_{axis}"
            setattr(self._glyph_actor.material, attr, idx)
        self._cur[axis] = idx

    def hide_axis(self, axis):
        """Hide all slices for *axis*.

        Parameters
        ----------
        axis : {"x", "y", "z"}
            Which axis's slice-visibility uniform to clear.
        """
        if self._glyph_actor is not None:
            setattr(self._glyph_actor.material, f"vis_{axis}", 0)
        self._cur[axis] = -1

    def show_axis(self, axis):
        """Enable axis visibility.

        Parameters
        ----------
        axis : {"x", "y", "z"}
            Which axis's slice-visibility uniform to set.
        """
        if self._glyph_actor is not None:
            setattr(self._glyph_actor.material, f"vis_{axis}", 1)

    def set_scale(self, new_scale):
        """Update scale on the actor.

        Parameters
        ----------
        new_scale : float
            New uniform billboard size multiplier.
        """
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
        """Set opacity.

        Parameters
        ----------
        opacity : float
            Glyph opacity as a fraction, expected in ``[0, 1]``. Below
            ``1.0`` the material's ``alpha_mode`` switches to ``"blend"``.
        """
        self._opacity = float(opacity)
        a = self._glyph_actor
        if a is not None:
            a.material.opacity = float(opacity)
            a.material.alpha_mode = "blend" if opacity < 1.0 else "solid"


class SHGlyph3D(Visualization):
    """High-level ODF visualization: UI widgets, sync, and slice state.

    Wraps a single :class:`SHSlicer` and converts the shared, world-space
    ``state`` vector (synchronized across every Skyline visualization) into
    the per-axis slice positions the billboard shader expects.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    coeffs : ndarray, shape (X, Y, Z, C)
        SH coefficients per voxel.
    affine : ndarray, optional
        Voxel-to-world affine used to position slices in world coordinates.
        When ``None``, ``state``/slice positions are voxel indices instead.
    render_callback : callable, optional
        Callback used to request a render/update.
    scale : float, optional
        Per-glyph scale used only when ``affine`` is ``None``; otherwise
        the scale is derived from the affine's voxel sizes.
    l_max : int, optional
        Maximum SH order to shade.
    lut_res : int, optional
        Cube-map Hermite LUT resolution per face edge.
    basis_type : {"standard", "descoteaux", "descoteaux07"}, optional
        SH basis convention of ``coeffs``.
    color_type : {"orientation", "sign"}, optional
        Glyph coloring: direction-mapped hue, or a two-color sign split.
    mask : ndarray of bool, optional
        Boolean mask of valid voxels.
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
        basis_type="standard",
        color_type="orientation",
        mask=None,
        sync_callback=None,
    ):
        """Initialize the ODF visualization.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        coeffs : ndarray, shape (X, Y, Z, C)
            SH coefficients per voxel.
        affine : ndarray, optional
            Voxel-to-world affine used to position slices in world
            coordinates. When ``None``, ``state``/slice positions are
            voxel indices instead.
        render_callback : callable, optional
            Callback used to request a render/update.
        scale : float, optional
            Per-glyph scale used only when ``affine`` is ``None``;
            otherwise the scale is derived from the affine's voxel sizes.
        l_max : int, optional
            Maximum SH order to shade.
        lut_res : int, optional
            Cube-map Hermite LUT resolution per face edge.
        basis_type : {"standard", "descoteaux", "descoteaux07"}, optional
            SH basis convention of ``coeffs``.
        color_type : {"orientation", "sign"}, optional
            Glyph coloring: direction-mapped hue, or a two-color sign split.
        mask : ndarray of bool, optional
            Boolean mask of valid voxels.
        sync_callback : callable, optional
            Callback used to synchronize state across views.
        """
        self.affine = affine
        if self.affine is not None:
            default_scale = float(np.mean(affine_voxel_sizes(self.affine)))
        else:
            default_scale = float(scale)

        self.shape = coeffs.shape[:3]

        self._slicer = SHSlicer(
            coeffs,
            scale=default_scale,
            l_max=l_max,
            lut_res=lut_res,
            mask=mask,
            basis_type=basis_type,
            color_type=color_type,
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

        self._last_state = [-1, -1, -1]

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
        """Group actor to add to the scene; delegates to the slicer.

        Returns
        -------
        Group
            Parent group containing the billboard actor.
        """
        return self._slicer.actor

    def _populate_info(self):
        """Build the multi-line summary shown in the info panel.

        Returns
        -------
        str
            Dimensions, SH coefficient count and order, plus voxel sizes,
            voxel order, and affine when an affine is available.
        """
        info = f"Dimensions: {self.shape}"
        info += f"\nSH Coefficients: {self._slicer.n_coeffs}"
        info += f"\nSH Order: {self._slicer.l_max}"
        if self.affine is not None:
            info += "\n" + format_affine_info(self.affine)

        return info

    def _voxel_from_world_state(self, world_state):
        """Snap a world-space state vector to the nearest in-bounds voxel index.

        Parameters
        ----------
        world_state : array-like
            World-space state vector to map into voxel coordinates.

        Returns
        -------
        np.ndarray
            Integer voxel index, clipped to the volume bounds.
        """
        if self.affine is None:
            return np.clip(
                np.round(world_state).astype(int), 0, np.array(self.shape) - 1
            )
        voxel = apply_transformation(
            np.array([world_state], dtype=np.float32), np.linalg.inv(self.affine)
        )[0]
        return np.clip(np.round(voxel).astype(int), 0, np.array(self.shape) - 1)

    def set_slices(self):
        """Push the current ``state`` to the billboard material's slice uniforms.

        Snaps ``state`` to the nearest voxel (:meth:`_voxel_from_world_state`),
        then forward-transforms that voxel back to world space (when an
        affine is present) before writing it to each axis's
        ``active_slice_*`` uniform.  This mirrors how ``Peak3D`` derives its
        cross section, and pairs with the vertex shader's own snap-onto-plane
        logic to keep the rendered slice crisp for any affine, including
        rotated or axis-swapped ones.
        """
        voxel = self._voxel_from_world_state(self.state)
        if self.affine is not None:
            slice_state = apply_transformation(
                np.array([voxel], dtype=np.float32), self.affine
            )[0]
        else:
            slice_state = voxel.astype(float)
        for i, axis in enumerate(("x", "y", "z")):
            self._slicer.set_slice(axis, float(slice_state[i]))
            self._last_state[i] = self.state[i]

    def update_state(self, new_state):
        """Apply a synchronized world-space state from another visualization.

        Ignored when :attr:`_synchronize` is off (per-view slice sync toggle).

        Parameters
        ----------
        new_state : array-like
            New shared world-space (x, y, z) state; only the first 3
            components are used.
        """
        if self._synchronize:
            self.state = new_state[:3]
            self.apply_scene_op(self.set_slices)

    def set_slice_visibility(self):
        """Show/hide each axis's slice per :attr:`_slice_visibility`."""
        for i, axis in enumerate(("x", "y", "z")):
            if self._slice_visibility[i]:
                self._slicer.show_axis(axis)
                self._last_state[i] = self.state[i]
            else:
                self._slicer.hide_axis(axis)
                self._last_state[i] = -1

    def render_widgets(self):
        """Draw the sync toggle, scale/opacity controls, and per-axis sliders."""
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
        slider_bounds = slice_slider_bounds(self.shape, affine=self.affine)
        slider_state = slice_slider_values_from_state(self.state, affine=self.affine)
        slicers = []
        for axis, label in enumerate(axis_labels):
            min_bound, max_bound = slider_bounds[axis]
            slicers.append(
                (
                    thin_slider,
                    (label, slider_state[axis], min_bound, max_bound),
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
                slider_state[idx] = round(new)
                self.state = slice_state_from_slider_values(
                    slider_state, affine=self.affine
                )
                self.apply_scene_op(self.set_slices)
                if self._synchronize and self._sync_callback is not None:
                    self._sync_callback(self, self.state)
            self._slice_visibility[idx] = toggle
            self.apply_scene_op(self.set_slice_visibility)
            self._last_state[idx] = -1

        imgui.spacing()


if not has_fury_v2:
    create_shm_visualization = SHSlicer = SHGlyph3D = fury
