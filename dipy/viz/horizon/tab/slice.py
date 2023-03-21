import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import (HorizonTab, build_label, color_double_slider,
                                  color_single_slider)

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class SlicesTab(HorizonTab):
    def __init__(
        self, slice_actors, img_shape, img_value_ranges, adjusted_data
        ):
        
        self.__actors = slice_actors
        self.__name = 'Slices'
        
        self.__tab_id = 0
        self.__tab_ui = None
        
        self.__shape = img_shape
        self.__ranges = img_value_ranges
        self.__data = adjusted_data
        
        self.__slider_label_opacity = build_label(text='Opacity')
        
        opacity = 1
        
        length = 450
        lw = 3
        radius = 8
        fs = 16
        
        tt = '{ratio:.0%}'
        
        self.__slider_opacity = ui.LineSlider2D(
            initial_value=opacity, min_value=.0, max_value=1., length=length,
            line_width=lw, outer_radius=radius, font_size=fs, text_template=tt)
        
        color_single_slider(self.__slider_opacity)
        
        self.__slider_opacity.on_change = self.__change_opacity
        
        self.__slider_label_x = build_label(text='X Slice')
        self.__slider_label_y = build_label(text='Y Slice')
        self.__slider_label_z = build_label(text='Z Slice')
        
        tt = '{value:.0f}'
        
        self.__slider_slice_x = ui.LineSlider2D(
            initial_value=self.__shape[0] / 2, min_value=0,
            max_value=self.__shape[0] - 1, length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        self.__slider_slice_y = ui.LineSlider2D(
            initial_value=self.__shape[1] / 2, min_value=0,
            max_value=self.__shape[1] - 1, length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        self.__slider_slice_z = ui.LineSlider2D(
            initial_value=self.__shape[2] / 2, min_value=0,
            max_value=self.__shape[2] - 1, length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        color_single_slider(self.__slider_slice_x)
        color_single_slider(self.__slider_slice_y)
        color_single_slider(self.__slider_slice_z)
        
        self.__slider_slice_x.on_change = self.__change_slice_x
        self.__slider_slice_y.on_change = self.__change_slice_y
        self.__slider_slice_z.on_change = self.__change_slice_z
        
        self.__slider_label_intensities = build_label(text='Intensities')
        
        self.__slider_intensities = ui.LineDoubleSlider2D(
            initial_values=self.__ranges, min_value=self.__data.min(),
            max_value=self.__data.max(), length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        color_double_slider(self.__slider_intensities)
    
    def __change_opacity(self, slider):
        opacity = slider.value
        for slice in self.__actors:
            slice.GetProperty().SetOpacity(opacity)
    
    def __change_slice_x(self, slider):
        value = int(np.rint(slider.value))
        self.__actors[0].display_extent(
            value, value, 0, self.__shape[1] - 1, 0, self.__shape[2] - 1)
    
    def __change_slice_y(self, slider):
        value = int(np.rint(slider.value))
        self.__actors[1].display_extent(
            0, self.__shape[0] - 1, value, value, 0, self.__shape[2] - 1)
    
    def __change_slice_z(self, slider):
        value = int(np.rint(slider.value))
        self.__actors[2].display_extent(
            0, self.__shape[0] - 1, 0, self.__shape[1] - 1, value, value)
    
    def build(self, tab_id, tab_ui):
        self.__tab_id = tab_id
        self.__tab_ui = tab_ui
        
        x_pos = .02
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_opacity, (x_pos, .85))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_x, (x_pos, .62))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_y, (x_pos, .38))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_z, (x_pos, .15))
        
        x_pos = .1
        
        self.__tab_ui.add_element(
                self.__tab_id, self.__slider_opacity, (x_pos, .85))
        self.__tab_ui.add_element(
                self.__tab_id, self.__slider_slice_x, (x_pos, .62))
        self.__tab_ui.add_element(
                self.__tab_id, self.__slider_slice_y, (x_pos, .38))
        self.__tab_ui.add_element(
                self.__tab_id, self.__slider_slice_z, (x_pos, .15))
        
        x_pos = .52
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_intensities, (x_pos, .85))
        
        x_pos = .60
        
        self.__tab_ui.add_element(
                self.__tab_id, self.__slider_intensities, (x_pos, .85))
    
    @property
    def name(self):
        return self.__name
