"""Peak direction (PAM) slicers for Skyline."""

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import (
    create_numeric_input,
    render_group,
    thin_slider,
    toggle_button,
)
from dipy.viz.skyline.render.renderer import (
    Visualization,
    format_affine_info,
    slice_slider_bounds,
    slice_slider_values_from_state,
    slice_state_from_slider_values,
)

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
    from fury.actor import peaks_slicer, set_group_opacity
    from fury.transform import apply_transformation

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if has_imgui:
    imgui = imgui_bundle.imgui


def create_peak_visualization(
    input,
    idx,
    *,
    opacity=100,
    render_callback=None,
    sync_callabck=None,
):
    """Create a peak visualization from loaded PAM data.

    Parameters
    ----------
    input : tuple
        Tuple of the form ``(pam, filename)`` or ``(pam,)``, where ``pam``
        is a :class:`~dipy.direction.peaks.PeaksAndMetrics` instance.
    idx : int
        Index of the peak for naming purposes if filename is not provided.
    opacity : int, optional
        Initial opacity of the peak rendering, in percent ``[0, 100]``.
    render_callback : callable, optional
        Callback function to be called after rendering.
    sync_callabck : callable, optional
        Callback function to synchronize slice positions across visualizations.

    Returns
    -------
    Peak3D
        The created Peak3D object.

    Raises
    ------
    ValueError
        If ``input`` is not a tuple of length 1 or 2.
    """
    if not isinstance(input, tuple) or len(input) not in (1, 2):
        raise ValueError(
            "Input must be a tuple containing (pam, filename) or (pam,) "
            "for peak visualization."
        )

    if len(input) == 1:
        pam = input[0]
        filename = f"Peaks_{idx}"
    else:
        pam, filename = input

    peak_values = 1.0
    if pam.peak_values is not None:
        max = np.percentile(pam.peak_values, 99)
        peak_values = np.clip(pam.peak_values, 0, max)

    return Peak3D(
        filename,
        pam.peak_dirs,
        affine=pam.affine,
        peak_values=peak_values,
        opacity=opacity,
        render_callback=render_callback,
        sync_callabck=sync_callabck,
    )


class Peak3D(Visualization):
    """Represent a peak-direction (PAM) vector-field slicer in Skyline.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    peaks : ndarray, shape (X, Y, Z, N, 3) or (X, Y, Z, 3)
        Per-voxel peak directions rendered as a vector field.
    affine : ndarray, optional
        Voxel-to-world affine used to position slices in world coordinates.
    peak_values : ndarray or float, optional
        Per-peak magnitude scaling the rendered line length; a scalar
        value is applied uniformly to every peak.
    opacity : int, optional
        Initial value of the Opacity slider, in percent ``[0, 100]``. Stored
        on ``self.opacity`` but not applied to the actor until the Opacity
        slider is changed once in :meth:`render_widgets`.
    render_callback : callable, optional
        Callback used to request a render/update.
    sync_callabck : callable, optional
        Callback used to synchronize state across views.
    """

    def __init__(
        self,
        name,
        peaks,
        *,
        affine=None,
        peak_values=1.0,
        opacity=100,
        render_callback=None,
        sync_callabck=None,
    ):
        """Initialize the peak-direction (PAM) vector-field slicer.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        peaks : ndarray, shape (X, Y, Z, N, 3) or (X, Y, Z, 3)
            Per-voxel peak directions rendered as a vector field.
        affine : ndarray, optional
            Voxel-to-world affine used to position slices in world coordinates.
        peak_values : ndarray or float, optional
            Per-peak magnitude scaling the rendered line length; a scalar
            value is applied uniformly to every peak.
        opacity : int, optional
            Initial value of the Opacity slider, in percent ``[0, 100]``.
            Stored on ``self.opacity`` but not applied to the actor until
            the Opacity slider is changed once in :meth:`render_widgets`.
        render_callback : callable, optional
            Callback used to request a render/update.
        sync_callabck : callable, optional
            Callback used to synchronize state across views.
        """
        self.peaks = peaks
        self.affine = affine
        self.peak_values = peak_values
        self._scale = 1.0
        self.opacity = opacity
        self._synchronize = True
        self._sync_callabck = sync_callabck
        self._slice_visibility = [True, True, True]
        self._create_peak_actor()
        super().__init__(name, render_callback)

    def _create_peak_actor(self):
        """Build the peaks-slicer actor and derive its cross-section state.

        Called from :meth:`__init__` and again whenever the Scale slider
        changes, since the slicer actor must be rebuilt for a new scale.
        """
        self._slicer = peaks_slicer(
            self.peaks,
            affine=self.affine,
            peak_values=self.peak_values * self._scale,
            visibility=self._slice_visibility,
        )
        self.state = self._get_cross_section()
        self._cross_section_state = np.asarray(self.state, dtype=np.float32)
        lower_bounds = np.zeros(3)
        upper_bounds = np.array(self.peaks.shape[:3]) - 1
        if self.affine is not None:
            self.bounds = apply_transformation(
                np.array([lower_bounds, upper_bounds]), self.affine
            )
        else:
            self.bounds = np.asarray([lower_bounds, upper_bounds])
        self._cross_section_space = self._infer_cross_section_space()
        self._apply_cross_section_from_state()

    def _populate_info(self):
        """Build the multi-line summary shown in the info panel.

        Returns
        -------
        str
            Peaks array shape and dtype, plus affine details when available.
        """
        info = f"Peaks shape: {self.peaks.shape}\n"
        info += f"Peaks dtype: {self.peaks.dtype}\n"
        if self.affine is not None:
            info += format_affine_info(self.affine) + "\n"
        return info

    @property
    def actor(self):
        """Vector-field actor group backing this peak visualization.

        Returns
        -------
        Group
            Parent group of chunked vector-field actors rendered as the
            three orthogonal peak-direction slices.
        """
        return self._slicer

    def _get_cross_section(self):
        """Read the shared cross-section position off the slicer's first chunk.

        ``peaks_slicer`` returns a ``Group`` of chunked ``VectorField`` actors and
        only the chunks carry the ``cross_section`` property, so read it off the
        first chunk. Every chunk is kept at the same cross section.

        Returns
        -------
        np.ndarray
            Current cross-section position, in the same space (voxel or
            world) the slicer actor was last set to.
        """
        return np.asarray(self._slicer.children[0].cross_section, dtype=np.float32)

    def _set_cross_section(self, cross_section):
        """Propagate a cross-section position to every chunk of the slicer.

        Parameters
        ----------
        cross_section : array-like
            Cross section to propagate to every chunk of the slicer.
        """
        for chunk in self._slicer.children:
            chunk.cross_section = cross_section

    def _infer_cross_section_space(self):
        """Determine whether the current cross section is voxel or world space.

        Returns
        -------
        str
            ``"voxel"`` when no affine is set, or when the cross section
            reported by ``peaks_slicer`` is closer to the voxel-space
            volume center than to its world-space counterpart; otherwise
            ``"world"``.
        """
        if self.affine is None:
            return "voxel"

        cross_section = self._get_cross_section()
        voxel_center = (np.array(self.peaks.shape[:3], dtype=np.float32) - 1.0) * 0.5
        world_center = apply_transformation(
            np.array([voxel_center], dtype=np.float32), self.affine
        )[0]

        world_dist = np.linalg.norm(cross_section - world_center)
        voxel_dist = np.linalg.norm(cross_section - voxel_center)
        return "world" if world_dist <= voxel_dist else "voxel"

    def _voxel_from_world_state(self, world_state):
        """Map a world-space state vector to a clipped voxel index.

        Parameters
        ----------
        world_state : array-like
            World-space state vector to map into local slice coordinates.

        Returns
        -------
        np.ndarray
            Voxel index nearest to ``world_state``, clipped to the volume
            bounds.
        """
        voxel_state = apply_transformation(
            np.array([world_state], dtype=np.float32), np.linalg.inv(self.affine)
        )[0]
        voxel_state = np.round(voxel_state).astype(np.int16)
        max_idx = np.array(self.peaks.shape[:3], dtype=np.int16) - 1
        return np.clip(voxel_state, 0, max_idx)

    def _apply_cross_section_from_state(self):
        """Push ``self.state`` to the slicer as a cross section.

        Converts ``self.state`` to voxel or world coordinates to match
        :attr:`_cross_section_space` before writing it to every chunk of
        the slicer via :meth:`_set_cross_section`.
        """
        if self.affine is None:
            voxel_state = np.round(self.state).astype(np.int16)
            self._cross_section_state = voxel_state.astype(np.float32)
            self._set_cross_section(voxel_state)
            return

        voxel_state = self._voxel_from_world_state(self.state)
        if self._cross_section_space == "world":
            world_state = apply_transformation(
                np.array([voxel_state], dtype=np.float32), self.affine
            )[0]
            self._cross_section_state = np.asarray(world_state, dtype=np.float32)
            self._set_cross_section(self._cross_section_state)
        else:
            self._cross_section_state = voxel_state.astype(np.float32)
            self._set_cross_section(voxel_state)

    def update_state(self, new_state):
        """Apply a synchronized state from another visualization.

        Parameters
        ----------
        new_state : array-like
            New synchronized state for this visualization.
        """
        if self._synchronize:
            self.state = np.asarray(new_state[:3], dtype=np.float32)
            self.apply_scene_op(self._apply_cross_section_from_state)

    def _set_opacity(self, opacity):
        """Apply an opacity fraction to every chunk of the peaks-slicer actor.

        Parameters
        ----------
        opacity : float
            Slice opacity, expected in ``[0, 1]``.
        """
        set_group_opacity(self._slicer, opacity)

    def _set_slice_visibility(self, visibility):
        """Apply per-axis slice visibility to every chunk's material.

        ``peaks_slicer`` returns a ``Group`` whose ``material`` is ``None``; the
        per-axis visibility flags live on the material of each chunk.

        Parameters
        ----------
        visibility : tuple(bool, bool, bool)
            Per-axis visibility flags for X/Y/Z slices.
        """
        for chunk in self._slicer.children:
            chunk.material.visibility = visibility

    def render_widgets(self):
        """Draw the sync toggle, scale, opacity, and per-axis slice controls."""
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
                self._scale = new_scale
                self.apply_scene_op(self._create_peak_actor)
                self.render()

        changed, new = thin_slider(
            "Opacity",
            self.opacity,
            0,
            100,
            value_type="int",
            text_format=".0f",
            value_unit="%",
            step=1,
        )
        if changed:
            self.opacity = new
            self.apply_scene_op(self._set_opacity, self.opacity / 100.0)

        imgui.spacing()
        axis_labels = ("X", "Y", "Z")
        slider_bounds = slice_slider_bounds(self.peaks.shape[:3], affine=self.affine)
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
                slider_state[idx] = float(new)
                self.state = slice_state_from_slider_values(
                    slider_state, affine=self.affine
                )
                self.apply_scene_op(self._apply_cross_section_from_state)
                if self._synchronize and self._sync_callabck is not None:
                    self._sync_callabck(self, self.state)
            self._slice_visibility[idx] = toggle
        self.apply_scene_op(self._set_slice_visibility, tuple(self._slice_visibility))

        imgui.spacing()


if not has_fury_v2:
    create_peak_visualization = Peak3D = fury
