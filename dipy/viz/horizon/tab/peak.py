from functools import partial
import numpy as np

from dipy.viz.horizon.tab import (HorizonTab, build_label, build_slider,
                                  build_radio_button, build_checkbox)


class PeaksTab(HorizonTab):
    def __init__(self, peak_actor):
        """Initialize Interaction tab for peaks visualization.

        Parameters
        ----------
        peak_actor : PeaksActor
            Horizon PeaksActor visualize peaks.
        """
        super().__init__()

        self._actor = peak_actor
        self._name = 'Peaks'

        self._tab_id = 0

        self._opacity_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._toggle_opacity)

        self._opacity_label, self._opacity = build_slider(
            initial_value=1.,
            max_value=1.,
            text_template='{ratio:.0%}',
            on_change=self._change_opacity,
            label='Opacity'
        )

        min_centers = self._actor.min_centers
        max_centers = self._actor.max_centers
        cross_section = self._actor.cross_section

        self._slice_x_label, self._slice_x = build_slider(
            initial_value=cross_section[0],
            min_value=min_centers[0],
            max_value=max_centers[0],
            text_template='{value:.0f}',
            label='X Slice'
        )
        self._change_slice_x = partial(
            self._change_slice, selected_slice=self._slice_x)
        self._adjust_slice_x = partial(self._change_slice_x, sync_slice=True)
        self._slice_x.obj.on_moving_slider = self._change_slice_x
        self._slice_x.obj.on_value_changed = self._adjust_slice_x

        self._slice_y_label, self._slice_y = build_slider(
            initial_value=cross_section[1],
            min_value=min_centers[1],
            max_value=max_centers[1],
            text_template='{value:.0f}',
            label='Y Slice'
        )
        self._change_slice_y = partial(
            self._change_slice, selected_slice=self._slice_y)
        self._adjust_slice_y = partial(self._change_slice_y, sync_slice=True)
        self._slice_y.obj.on_moving_slider = self._change_slice_y
        self._slice_y.obj.on_value_changed = self._adjust_slice_y

        self._slice_z_label, self._slice_z = build_slider(
            initial_value=cross_section[2],
            min_value=min_centers[2],
            max_value=max_centers[2],
            text_template='{value:.0f}',
            label='Z Slice'
        )
        self._change_slice_z = partial(
            self._change_slice, selected_slice=self._slice_z)
        self._adjust_slice_z = partial(self._change_slice_z, sync_slice=True)
        self._slice_z.obj.on_moving_slider = self._change_slice_z
        self._slice_z.obj.on_value_changed = self._adjust_slice_z

        low_ranges = self._actor.low_ranges
        high_ranges = self._actor.high_ranges

        self._range_x_label, self._range_x = build_slider(
            initial_value=(low_ranges[0], high_ranges[0]),
            min_value=low_ranges[0], max_value=high_ranges[0],
            text_template='{value:.0f}',
            label='X Range',
            is_double_slider=True
        )
        self._change_range_x = partial(
            self._change_range, selected_range=self._range_x)
        self._range_x.obj.on_change = self._change_range_x

        self._range_y_label, self._range_y = build_slider(
            initial_value=(low_ranges[1], high_ranges[1]),
            min_value=low_ranges[1], max_value=high_ranges[1],
            text_template='{value:.0f}',
            label='Y Range',
            is_double_slider=True
        )
        self._change_range_y = partial(
            self._change_range, selected_range=self._range_y)
        self._range_y.obj.on_change = self._change_range_y

        self._range_z_label, self._range_z = build_slider(
            initial_value=(low_ranges[2], high_ranges[2]),
            min_value=low_ranges[2], max_value=high_ranges[2],
            text_template='{value:.0f}',
            label='Z Range',
            is_double_slider=True
        )
        self._change_range_z = partial(
            self._change_range, selected_range=self._range_z)
        self._range_z.obj.on_change = self._change_range_z

        self._view_mode_label = build_label(
            text='View Mode', is_horizon_label=True)

        self._view_modes = ['Cross section', 'Range']
        self._view_mode_toggler = build_radio_button(
            self._view_modes, [self._view_modes[0]],
            padding=1.5, on_change=self._toggle_view_mode)

        self.register_elements(
            self._opacity_label, self._opacity_toggle, self._opacity,
            self._slice_x_label, self._slice_x,
            self._slice_y_label, self._slice_y,
            self._slice_z_label, self._slice_z,
            self._range_x_label, self._range_x,
            self._range_y_label, self._range_y,
            self._range_z_label, self._range_z,
            self._view_mode_label, self._view_mode_toggler,
        )

    def _toggle_opacity(self, checkbox):
        """Toggle Opacity of the peaks actor.

        Parameters
        ----------
        checkbox : Checkbox
            FURY checkbox UI element.
        """
        if '' in checkbox.checked_labels:
            self._opacity.obj.value = 1
        else:
            self._opacity.obj.value = 0

    def _change_opacity(self, slider):
        """Update opacity of the peaks actor by adjusting the slider to
        suitable value.

        Parameters
        ----------
        slider : LineSlider2D
            FURY slider UI element.
        """
        self._actor.global_opacity = slider.value

    def _change_range(self, slider, selected_range):
        """Update the range of peaks actor by adjusting the double slider.
        Only usable in Range view mode.

        Parameters
        ----------
        slider : LineDoubleSlider2D
            FURY two side slider UI element.
        selected_range : HorizonUIElement
            Selected horizon ui element intended to change.
        """
        selected_range.selected_value = (
            slider.left_disk_value, slider.right_disk_value)
        self._actor.display_extent(
            self._range_x.selected_value[0], self._range_x.selected_value[1],
            self._range_y.selected_value[0], self._range_y.selected_value[1],
            self._range_z.selected_value[0], self._range_z.selected_value[1],
        )

    def _change_slice(
            self, slider, selected_slice, sync_slice=False):
        """Update the slice value of peaks actor by adjusting the slider. Only
        usable in Cross Section view mode.

        Parameters
        ----------
        slider : LineSlider2D
            FURY slider UI element.
        selected_slice : HorizonUIElement
            Selected horizon ui element intended to change.
        sync_slice : bool, optional
            Whether the slice is getting synchronized some other change,
            by default False
        """
        value = int(np.rint(slider.value))
        selected_slice.selected_value = value

        if not sync_slice:
            self.on_slice_change(
                self._tab_id,
                self._slice_x.selected_value,
                self._slice_y.selected_value,
                self._slice_z.selected_value,
            )

        if self._view_mode_toggler.selected_value[0] == self._view_modes[0]:
            self._actor.display_cross_section(
                self._slice_x.selected_value,
                self._slice_y.selected_value,
                self._slice_z.selected_value,
            )

    def _show_cross_section(self):
        """Show Cross Section view mode. Hide the Range sliders and labels.
        """
        self.hide(
            self._range_x_label, self._range_x,
            self._range_y_label, self._range_y,
            self._range_z_label, self._range_z,
        )
        self.show(
            self._slice_x_label, self._slice_x,
            self._slice_y_label, self._slice_y,
            self._slice_z_label, self._slice_z,
        )
        self._change_slice(self._slice_x.obj, self._slice_x, sync_slice=True)

    def _show_range(self):
        """Show Range view mode. Hide Cross Section sliders and labels.
        """
        self.hide(
            self._slice_x_label, self._slice_x,
            self._slice_y_label, self._slice_y,
            self._slice_z_label, self._slice_z,
        )
        self.show(
            self._range_x_label, self._range_x,
            self._range_y_label, self._range_y,
            self._range_z_label, self._range_z,
        )

    def _toggle_view_mode(self, radio):
        self._view_mode_toggler.selected_value = radio.checked_labels
        if radio.checked_labels[0] == self._view_modes[0]:
            self._actor.display_cross_section(
                self._actor.cross_section[0], self._actor.cross_section[1],
                self._actor.cross_section[2])
            self._show_cross_section()
        else:
            self._actor.display_extent(
                self._actor.low_ranges[0], self._actor.high_ranges[0],
                self._actor.low_ranges[1], self._actor.high_ranges[1],
                self._actor.low_ranges[2], self._actor.high_ranges[2])
            self._show_range()

    def on_tab_selected(self):
        """Trigger when tab becomes active.
        """
        self._toggle_view_mode(self._view_mode_toggler.obj)

    def update_slices(self, x_slice, y_slice, z_slice):
        """Updates slicer positions.

        Parameters
        ----------
        x_slice: float
            x-value where the slicer should be placed
        y_slice: float
            y-value where the slicer should be placed
        z_slice: float
            z-value where the slicer should be placed
        """
        if not self._slice_x.obj.value == x_slice:
            self._slice_x.obj.value = x_slice

        if not self._slice_y.obj.value == y_slice:
            self._slice_y.obj.value = y_slice

        if not self._slice_z.obj.value == z_slice:
            self._slice_z.obj.value = z_slice

    def build(self, tab_id, tab_ui):
        """Build all the elements under the tab.

        Parameters
        ----------
        tab_id : int
            Id of the tab.
        tab_ui : TabUI
            FURY TabUI object for tabs panel.

        Notes
        -----
        tab_ui will removed once every all tabs adapt new build architecture.
        """
        self._tab_id = tab_id

        x_pos = .02
        self._opacity_toggle.position = (x_pos, .85)

        x_pos = .04
        self._opacity_label.position = (x_pos, .85)
        self._slice_x_label.position = (x_pos, .62)
        self._slice_y_label.position = (x_pos, .38)
        self._slice_z_label.position = (x_pos, .15)
        self._range_x_label.position = (x_pos, .62)
        self._range_y_label.position = (x_pos, .38)
        self._range_z_label.position = (x_pos, .15)

        x_pos = .10
        self._opacity.position = (x_pos, .85)
        self._slice_x.position = (x_pos, .62)
        self._slice_y.position = (x_pos, .38)
        self._slice_z.position = (x_pos, .15)

        x_pos = .105
        self._range_x.position = (x_pos, .66)
        self._range_y.position = (x_pos, .42)
        self._range_z.position = (x_pos, .19)

        x_pos = .52
        self._view_mode_label.position = (x_pos, .85)

        x_pos = .62
        self._view_mode_toggler.position = (x_pos, .80)

        cross_section = self._actor.cross_section
        self._actor.display_cross_section(
            cross_section[0], cross_section[1], cross_section[2])

    @property
    def name(self):
        """Name of the tab.

        Returns
        -------
        str
        """
        return self._name

    @property
    def actors(self):
        """actors controlled by tab.

        Returns
        -------
        list
            List of actors.
        """
        return [self._actor]

    @property
    def tab_id(self):
        """Id of the tab. Reference for Tab Manager to identify the tab.

        Returns
        -------
        int
        """
        return self._tab_id
