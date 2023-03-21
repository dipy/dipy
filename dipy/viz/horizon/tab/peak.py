import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import HorizonTab, build_label, color_slider

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class PeaksTab(HorizonTab):
    def __init__(self, peak_actor):
        self.__actor = peak_actor
        self.__name = 'Peaks'
        
        self.__tab_id = 0
        self.__tab_ui = None
        
        self.__interaction_mode_label = build_label(
            text='Planes', font_size=20, bold=True)
        
        self.__interaction_mode_label.actor.AddObserver(
            'LeftButtonPressEvent', self.__change_interaction_mode_callback,
            1.)
        
        self.__slider_label_x = build_label(text='X Slice')
        self.__slider_label_y = build_label(text='Y Slice')
        self.__slider_label_z = build_label(text='Z Slice')
        
        # Initializing sliders
        min_centers = self.__actor.min_centers
        max_centers = self.__actor.max_centers
        
        length = 185
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
        
        color_slider(self.__slider_slice_x)
        color_slider(self.__slider_slice_y)
        color_slider(self.__slider_slice_z)
        
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
        
        self.__slider_range_x.on_change = self.__change_range_x
        self.__slider_range_y.on_change = self.__change_range_y
        self.__slider_range_z.on_change = self.__change_range_z
    
    def __add_cross_section_sliders(self, x_pos=.33):
        if self.__tab_ui is not None:
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_slice_x, (x_pos, .68))
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_slice_y, (x_pos, .43))
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_slice_z, (x_pos, .18))
        else:
            raise ValueError('')
    
    def __add_range_sliders(self, x_pos=.33):
        if self.__tab_ui is not None:
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_range_x, (x_pos, .68))
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_range_y, (x_pos, .43))
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_range_z, (x_pos, .18))
        else:
            raise ValueError('')
    
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
    
    def __change_interaction_mode_callback(self, obj, event):
        if self.__actor.is_range:
            self.__interaction_mode_label.message = 'Planes'
            self.__actor.display_cross_section(
                self.__actor.cross_section[0], self.__actor.cross_section[1],
                self.__actor.cross_section[2])
            #scene.clear()
            self.__remove_range_sliders()
            self.__add_cross_section_sliders()
            #scene.add(self.__actor)
            #scene.add(panel)
        else:
            self.__interaction_mode_label.message = 'Range'
            self.__actor.display_extent(
                self.__actor.low_ranges[0], self.__actor.high_ranges[0],
                self.__actor.low_ranges[1], self.__actor.high_ranges[1],
                self.__actor.low_ranges[2], self.__actor.high_ranges[2])
            #scene.clear()
            self.__remove_cross_section_sliders()
            self.__add_range_sliders()
            #scene.add(self.__actor)
            #scene.add(panel)
        #iren.Render()
    
    def __remove_cross_section_sliders(self):
        if self.__tab_ui is not None:
            self.__tab_ui.remove_element(self.__tab_id, self.__slider_slice_x)
            self.__tab_ui.remove_element(self.__tab_id, self.__slider_slice_y)
            self.__tab_ui.remove_element(self.__tab_id, self.__slider_slice_z)
        else:
            raise ValueError('')
    
    def __remove_range_sliders(self):
        if self.__tab_ui is not None:
            self.__tab_ui.remove_element(self.__tab_id, self.__slider_range_x)
            self.__tab_ui.remove_element(self.__tab_id, self.__slider_range_y)
            self.__tab_ui.remove_element(self.__tab_id, self.__slider_range_z)
        else:
            raise ValueError('')
    
    def build(self, tab_id, tab_ui):
        self.__tab_id = tab_id
        self.__tab_ui = tab_ui
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__interaction_mode_label, (.03, .85))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_x, (.06, .68))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_y, (.06, .43))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_z, (.06, .18))
        
        # Default view of Peak Actor is range
        # self.__add_range_sliders()
        
        # Changing initial view to cross section
        cross_section = self.__actor.cross_section
        self.__actor.display_cross_section(
            cross_section[0], cross_section[1], cross_section[2])
        self.__add_cross_section_sliders()
    
    @property
    def name(self):
        return self.__name
