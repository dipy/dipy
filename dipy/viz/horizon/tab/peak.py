from functools import partial
import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import (HorizonTab, build_label, color_double_slider,
                                  color_single_slider)
from dipy.viz.horizon.tab.base import build_checkbox, build_slider

fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")

if has_fury:
    from fury import ui


class PeaksTab(HorizonTab):
    def __init__(self, peak_actor):
        self._actor = peak_actor
        self._name = 'Peaks'

        self._tab_id = 0
        self._tab_ui = None

        self._toggler_label_view_mode = build_label(text='View Mode')

        """
        self._interaction_mode_label.actor.AddObserver(
            'LeftButtonPressEvent', self._change_interaction_mode_callback,
            1.)
        """

        self._view_modes = {'Cross section': 0, 'Range': 1}
        self._view_mode_toggler = ui.RadioButton(
            list(self._view_modes), ['Cross section'], padding=1.5,
            font_size=16, font_family='Arial')

        self._view_mode_toggler.on_change = self._toggle_view_mode

        self._slider_label_x = build_label(text='X Slice')
        self._slider_label_y = build_label(text='Y Slice')
        self._slider_label_z = build_label(text='Z Slice')

        # Initializing sliders
        min_centers = self._actor.min_centers
        max_centers = self._actor.max_centers

        length = 450
        lw = 3
        radius = 8
        fs = 16
        tt = '{value:.0f}'

        # Initializing cross section sliders
        cross_section = self._actor.cross_section

        self._slider_slice_x = ui.LineSlider2D(
            initial_value=cross_section[0], min_value=min_centers[0],
            max_value=max_centers[0], length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)

        self._slider_slice_y = ui.LineSlider2D(
            initial_value=cross_section[1], min_value=min_centers[1],
            max_value=max_centers[1], length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)

        self._slider_slice_z = ui.LineSlider2D(
            initial_value=cross_section[2], min_value=min_centers[2],
            max_value=max_centers[2], length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)

        color_single_slider(self._slider_slice_x)
        color_single_slider(self._slider_slice_y)
        color_single_slider(self._slider_slice_z)

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

        self._adjust_slice_x = partial(self._change_slice_x, sync_slice=True)
        self._slider_slice_x.on_moving_slider = self._change_slice_x
        self._slider_slice_x.on_value_changed = self._adjust_slice_x

        self._adjust_slice_y = partial(self._change_slice_y, sync_slice=True)
        self._slider_slice_y.on_moving_slider = self._change_slice_y
        self._slider_slice_y.on_value_changed = self._adjust_slice_y

        self._adjust_slice_z = partial(self._change_slice_z, sync_slice=True)
        self._slider_slice_z.on_moving_slider = self._change_slice_z
        self._slider_slice_z.on_value_changed = self._adjust_slice_z

        # Initializing cross section sliders
        low_ranges = self._actor.low_ranges
        high_ranges = self._actor.high_ranges

        self._slider_range_x = ui.LineDoubleSlider2D(
            line_width=lw, outer_radius=radius, length=length,
            initial_values=(low_ranges[0], high_ranges[0]),
            min_value=low_ranges[0], max_value=high_ranges[0], font_size=fs,
            text_template=tt)

        self._slider_range_y = ui.LineDoubleSlider2D(
            line_width=lw, outer_radius=radius, length=length,
            initial_values=(low_ranges[1], high_ranges[1]),
            min_value=low_ranges[1], max_value=high_ranges[1], font_size=fs,
            text_template=tt)

        self._slider_range_z = ui.LineDoubleSlider2D(
            line_width=lw, outer_radius=radius, length=length,
            initial_values=(low_ranges[2], high_ranges[2]),
            min_value=low_ranges[2], max_value=high_ranges[2], font_size=fs,
            text_template=tt)

        color_double_slider(self._slider_range_x)
        color_double_slider(self._slider_range_y)
        color_double_slider(self._slider_range_z)

        self._slider_range_x.on_change = self._change_range_x
        self._slider_range_y.on_change = self._change_range_y
        self._slider_range_z.on_change = self._change_range_z

    def _toggle_opacity(self, checkbox):
        if '' in checkbox.checked_labels:
            self._opacity.obj.value = 1
        else:
            self._opacity.obj.value = 0

    def _change_opacity(self, slider):
        self._actor.global_opacity = slider.value

    def _add_cross_section_sliders(self, x_pos=.12):
        self._tab_ui.add_element(
            self._tab_id, self._slider_slice_x, (x_pos, .62))
        self._tab_ui.add_element(
            self._tab_id, self._slider_slice_y, (x_pos, .38))
        self._tab_ui.add_element(
            self._tab_id, self._slider_slice_z, (x_pos, .15))

    def _add_range_sliders(self, x_pos=.12):
        self._tab_ui.add_element(
            self._tab_id, self._slider_range_x, (x_pos, .62))
        self._tab_ui.add_element(
            self._tab_id, self._slider_range_y, (x_pos, .38))
        self._tab_ui.add_element(
            self._tab_id, self._slider_range_z, (x_pos, .15))

    def _change_range_x(self, slider):
        val1 = slider.left_disk_value
        val2 = slider.right_disk_value
        lr = self._actor.low_ranges
        hr = self._actor.high_ranges
        self._actor.display_extent(val1, val2, lr[1], hr[1], lr[2], hr[2])

    def _change_range_y(self, slider):
        val1 = slider.left_disk_value
        val2 = slider.right_disk_value
        lr = self._actor.low_ranges
        hr = self._actor.high_ranges
        self._actor.display_extent(lr[0], hr[0], val1, val2, lr[2], hr[2])

    def _change_range_z(self, slider):
        val1 = slider.left_disk_value
        val2 = slider.right_disk_value
        lr = self._actor.low_ranges
        hr = self._actor.high_ranges
        self._actor.display_extent(lr[0], hr[0], lr[1], hr[1], val1, val2)

    def _change_slice_x(self, slider, sync_slice=False):
        value = int(np.rint(slider.value))
        if not sync_slice:
            self.on_slice_change(
                self._tab_id,
                value,
                int(np.rint(self._slider_slice_y.value)),
                int(np.rint(self._slider_slice_z.value)),
            )
        cs = self._actor.cross_section
        self._actor.display_cross_section(value, cs[1], cs[2])

    def _change_slice_y(self, slider, sync_slice=False):
        value = int(np.rint(slider.value))
        if not sync_slice:
            self.on_slice_change(
                self._tab_id,
                int(np.rint(self._slider_slice_x.value)),
                value,
                int(np.rint(self._slider_slice_z.value)),
            )
        cs = self._actor.cross_section
        self._actor.display_cross_section(cs[0], value, cs[2])

    def _change_slice_z(self, slider, sync_slice=False):
        value = int(np.rint(slider.value))
        if not sync_slice:
            self.on_slice_change(
                self._tab_id,
                int(np.rint(self._slider_slice_x.value)),
                int(np.rint(self._slider_slice_y.value)),
                value,
            )
        cs = self._actor.cross_section
        self._actor.display_cross_section(cs[0], cs[1], value)

    def _hide_cross_section_sliders(self):
        self._slider_slice_x.set_visibility(False)
        self._slider_slice_y.set_visibility(False)
        self._slider_slice_z.set_visibility(False)

    def _hide_range_sliders(self):
        self._slider_range_x.set_visibility(False)
        self._slider_range_y.set_visibility(False)
        self._slider_range_z.set_visibility(False)

    def _show_cross_section_sliders(self):
        self._slider_slice_x.set_visibility(True)
        self._slider_slice_y.set_visibility(True)
        self._slider_slice_z.set_visibility(True)

    def _show_range_sliders(self):
        self._slider_range_x.set_visibility(True)
        self._slider_range_y.set_visibility(True)
        self._slider_range_z.set_visibility(True)

    def _toggle_view_mode(self, radio):
        view_mode = self._view_modes[radio.checked_labels[0]]
        # view_mode==0: Cross section - view_mode==1: Range
        if view_mode:
            self._actor.display_extent(
                self._actor.low_ranges[0], self._actor.high_ranges[0],
                self._actor.low_ranges[1], self._actor.high_ranges[1],
                self._actor.low_ranges[2], self._actor.high_ranges[2])
            self._hide_cross_section_sliders()
            self._show_range_sliders()
        else:
            self._actor.display_cross_section(
                self._actor.cross_section[0], self._actor.cross_section[1],
                self._actor.cross_section[2])
            self._hide_range_sliders()
            self._show_cross_section_sliders()

    def on_tab_selected(self):
        """Trigger when tab becomes active.
        """
        view_mode = self._view_modes[self._view_mode_toggler.checked_labels[0]]

        if view_mode:
            self._hide_cross_section_sliders()
        else:
            self._hide_range_sliders()

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
        if not self._slider_slice_x.value == x_slice:
            self._slider_slice_x.value = x_slice

        if not self._slider_slice_y.value == y_slice:
            self._slider_slice_y.value = y_slice

        if not self._slider_slice_z.value == z_slice:
            self._slider_slice_z.value = z_slice

    def build(self, tab_id, tab_ui):
        self._tab_id = tab_id
        self._tab_ui = tab_ui

        x_pos = .02

        self._tab_ui.add_element(
            self._tab_id, self._toggler_label_view_mode, (x_pos, .85))
        self._tab_ui.add_element(
            self._tab_id, self._slider_label_x, (x_pos, .62))
        self._tab_ui.add_element(
            self._tab_id, self._slider_label_y, (x_pos, .38))
        self._tab_ui.add_element(
            self._tab_id, self._slider_label_z, (x_pos, .15))

        x_pos = .12

        self._tab_ui.add_element(
            self._tab_id, self._view_mode_toggler, (x_pos, .80))

        y_pos = .85
        self._tab_ui.add_element(
            self._tab_id, self._opacity_toggle.obj, (.52, y_pos))
        self._tab_ui.add_element(
            self._tab_id, self._opacity_label.obj, (.55, y_pos))
        self._tab_ui.add_element(
            self._tab_id, self._opacity.obj, (.60, y_pos))

        # Default view of Peak Actor is range
        self._add_range_sliders()
        self._hide_range_sliders()

        # Changing initial view to cross section
        cross_section = self._actor.cross_section
        self._actor.display_cross_section(
            cross_section[0], cross_section[1], cross_section[2])
        self._add_cross_section_sliders()

    @property
    def name(self):
        return self._name

    @property
    def actors(self):
        return [self._actor]

    @property
    def tab_id(self):
        return self._tab_id
