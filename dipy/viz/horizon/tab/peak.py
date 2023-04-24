import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import (HorizonTab, build_label, color_double_slider,
                                  color_single_slider)

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class PeaksTab(HorizonTab):
    def __init__(self, peak_actor):
        self.__actor = peak_actor
        self.__name = 'Peaks'
        
        self.__tab_id = 0
        self.__tab_ui = None
        
        self.__toggler_label_view_mode = build_label(text='View Mode')
        
        """
        self.__interaction_mode_label.actor.AddObserver(
            'LeftButtonPressEvent', self.__change_interaction_mode_callback,
            1.)
        """
        
        self.__view_modes = {'Cross section': 0, 'Range': 1}
        self.__view_mode_toggler = ui.RadioButton(
            list(self.__view_modes), ['Cross section'], padding=1.5,
            font_size=16, font_family='Arial')
        
        self.__view_mode_toggler.on_change = self.__toggle_view_mode
        
        self.__slider_label_x = build_label(text='X Slice')
        self.__slider_label_y = build_label(text='Y Slice')
        self.__slider_label_z = build_label(text='Z Slice')
        
        # Initializing sliders
        min_centers = self.__actor.min_centers
        max_centers = self.__actor.max_centers
        
        length = 450
        lw = 3
        radius = 8
        fs = 16
        tt = '{value:.0f}'
        
        # Initializing cross section sliders
        cross_section = self.__actor.cross_section
        
        self.__slider_slice_x = ui.LineSlider2D(
            initial_value=cross_section[0], min_value=min_centers[0],
            max_value=max_centers[0], length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        self.__slider_slice_y = ui.LineSlider2D(
            initial_value=cross_section[1], min_value=min_centers[1],
            max_value=max_centers[1], length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        self.__slider_slice_z = ui.LineSlider2D(
            initial_value=cross_section[2], min_value=min_centers[2],
            max_value=max_centers[2], length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        color_single_slider(self.__slider_slice_x)
        color_single_slider(self.__slider_slice_y)
        color_single_slider(self.__slider_slice_z)
        
        self.__slider_slice_x.on_change = self.__change_slice_x
        self.__slider_slice_y.on_change = self.__change_slice_y
        self.__slider_slice_z.on_change = self.__change_slice_z
        
        # Initializing cross section sliders
        low_ranges = self.__actor.low_ranges
        high_ranges = self.__actor.high_ranges
        
        self.__slider_range_x = ui.LineDoubleSlider2D(
            line_width=lw, outer_radius=radius, length=length,
            initial_values=(low_ranges[0], high_ranges[0]),
            min_value=min_centers[0], max_value=max_centers[0], font_size=fs,
            text_template=tt)
        
        self.__slider_range_y = ui.LineDoubleSlider2D(
            line_width=lw, outer_radius=radius, length=length,
            initial_values=(low_ranges[1], high_ranges[1]),
            min_value=min_centers[1], max_value=max_centers[1], font_size=fs,
            text_template=tt)
        
        self.__slider_range_z = ui.LineDoubleSlider2D(
            line_width=lw, outer_radius=radius, length=length,
            initial_values=(low_ranges[2], high_ranges[2]),
            min_value=min_centers[2], max_value=max_centers[2], font_size=fs,
            text_template=tt)
        
        color_double_slider(self.__slider_range_x)
        color_double_slider(self.__slider_range_y)
        color_double_slider(self.__slider_range_z)
        
        self.__slider_range_x.on_change = self.__change_range_x
        self.__slider_range_y.on_change = self.__change_range_y
        self.__slider_range_z.on_change = self.__change_range_z
    
    def __add_cross_section_sliders(self, x_pos=.12):
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_slice_x, (x_pos, .62))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_slice_y, (x_pos, .38))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_slice_z, (x_pos, .15))
    
    def __add_range_sliders(self, x_pos=.12):
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_range_x, (x_pos, .62))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_range_y, (x_pos, .38))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_range_z, (x_pos, .15))
    
    def __change_range_x(self, slider):
        val1 = slider.left_disk_value
        val2 = slider.right_disk_value
        lr = self.__actor.low_ranges
        hr = self.__actor.high_ranges
        self.__actor.display_extent(val1, val2, lr[1], hr[1], lr[2], hr[2])
    
    def __change_range_y(self, slider):
        val1 = slider.left_disk_value
        val2 = slider.right_disk_value
        lr = self.__actor.low_ranges
        hr = self.__actor.high_ranges
        self.__actor.display_extent(lr[0], hr[0], val1, val2, lr[2], hr[2])
    
    def __change_range_z(self, slider):
        val1 = slider.left_disk_value
        val2 = slider.right_disk_value
        lr = self.__actor.low_ranges
        hr = self.__actor.high_ranges
        self.__actor.display_extent(lr[0], hr[0], lr[1], hr[1], val1, val2)
    
    def __change_slice_x(self, slider):
        value = int(np.rint(slider.value))
        cs = self.__actor.cross_section
        self.__actor.display_cross_section(value, cs[1], cs[2])
    
    def __change_slice_y(self, slider):
        value = int(np.rint(slider.value))
        cs = self.__actor.cross_section
        self.__actor.display_cross_section(cs[0], value, cs[2])
    
    def __change_slice_z(self, slider):
        value = int(np.rint(slider.value))
        cs = self.__actor.cross_section
        self.__actor.display_cross_section(cs[0], cs[1], value)
    
    def __hide_cross_section_sliders(self):
        self.__slider_slice_x.set_visibility(False)
        self.__slider_slice_y.set_visibility(False)
        self.__slider_slice_z.set_visibility(False)
    
    def __hide_range_sliders(self):
        self.__slider_range_x.set_visibility(False)
        self.__slider_range_y.set_visibility(False)
        self.__slider_range_z.set_visibility(False)
    
    def __show_cross_section_sliders(self):
        self.__slider_slice_x.set_visibility(True)
        self.__slider_slice_y.set_visibility(True)
        self.__slider_slice_z.set_visibility(True)
    
    def __show_range_sliders(self):
        self.__slider_range_x.set_visibility(True)
        self.__slider_range_y.set_visibility(True)
        self.__slider_range_z.set_visibility(True)
    
    def __toggle_view_mode(self, radio):
        view_mode = self.__view_modes[radio.checked_labels[0]]
        # view_mode==0: Cross section - view_mode==1: Range
        if view_mode:
            self.__actor.display_extent(
                self.__actor.low_ranges[0], self.__actor.high_ranges[0],
                self.__actor.low_ranges[1], self.__actor.high_ranges[1],
                self.__actor.low_ranges[2], self.__actor.high_ranges[2])
            self.__hide_cross_section_sliders()
            self.__show_range_sliders()
        else:
            self.__actor.display_cross_section(
                self.__actor.cross_section[0], self.__actor.cross_section[1],
                self.__actor.cross_section[2])
            self.__hide_range_sliders()
            self.__show_cross_section_sliders()
    
    def build(self, tab_id, tab_ui):
        self.__tab_id = tab_id
        self.__tab_ui = tab_ui
        
        x_pos = .02
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__toggler_label_view_mode, (x_pos, .85))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_x, (x_pos, .62))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_y, (x_pos, .38))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_z, (x_pos, .15))
        
        x_pos=.12
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__view_mode_toggler, (x_pos, .80))
        
        # Default view of Peak Actor is range
        self.__add_range_sliders()
        self.__hide_range_sliders()
        
        # Changing initial view to cross section
        cross_section = self.__actor.cross_section
        self.__actor.display_cross_section(
            cross_section[0], cross_section[1], cross_section[2])
        self.__add_cross_section_sliders()
    
    @property
    def name(self):
        return self.__name
