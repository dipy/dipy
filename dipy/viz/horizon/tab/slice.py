import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.loader import replace_volume_slice_actors
from dipy.viz.horizon.tab import (HorizonTab, build_label, color_double_slider,
                                  color_single_slider)

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import colormap, ui
    from fury.data import read_viz_icons


class SlicesTab(HorizonTab):
    def __init__(self, slices_loader):
        
        self.__actors = slices_loader.slice_actors
        self.__name = 'Slices'
        
        self.__tab_id = 0
        self.__tab_ui = None
        self.__global_memory = None
        
        self.__data_shape = slices_loader.data_shape
        self.__min_intensity = slices_loader.intensities_range[0]
        self.__max_intensity = slices_loader.intensities_range[1]
        
        self.__slider_label_opacity = build_label(text='Opacity')
        
        opacity = 1
        
        length = 450
        lw = 3
        radius = 8
        fs = 16
        
        tt = '{ratio:.0%}'
        
        self.__slider_opacity = ui.LineSlider2D(
            initial_value=opacity, max_value=1., length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        color_single_slider(self.__slider_opacity)
        
        self.__slider_opacity.on_change = self.__change_opacity
        
        self.__slider_label_x = build_label(text='X Slice')
        self.__slider_label_y = build_label(text='Y Slice')
        self.__slider_label_z = build_label(text='Z Slice')
        
        tt = '{value:.0f}'
        
        self.__slider_slice_x = ui.LineSlider2D(
            initial_value=self.__data_shape[0] / 2, min_value=0,
            max_value=self.__data_shape[0] - 1, length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        self.__slider_slice_y = ui.LineSlider2D(
            initial_value=self.__data_shape[1] / 2, min_value=0,
            max_value=self.__data_shape[1] - 1, length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        self.__slider_slice_z = ui.LineSlider2D(
            initial_value=self.__data_shape[2] / 2, min_value=0,
            max_value=self.__data_shape[2] - 1, length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        color_single_slider(self.__slider_slice_x)
        color_single_slider(self.__slider_slice_y)
        color_single_slider(self.__slider_slice_z)
        
        self.__slider_slice_x.on_change = self.__change_slice_x
        self.__slider_slice_y.on_change = self.__change_slice_y
        self.__slider_slice_z.on_change = self.__change_slice_z
        
        icon_files = []
        icon_files.append(('minus', read_viz_icons(fname='minus.png')))
        icon_files.append(('plus', read_viz_icons(fname='plus.png')))
        
        self.__button_slice_x = ui.Button2D(
            icon_fnames=icon_files, size=(25, 25))
        self.__button_slice_y = ui.Button2D(
            icon_fnames=icon_files, size=(25, 25))
        self.__button_slice_z = ui.Button2D(
            icon_fnames=icon_files, size=(25, 25))
        
        self.__slice_x_visibility = True
        self.__slice_y_visibility = True
        self.__slice_z_visibility = True
        
        self.__button_slice_x.on_left_mouse_button_clicked = (
            self.__change_slice_x_visibility)
        self.__button_slice_y.on_left_mouse_button_clicked = (
            self.__change_slice_y_visibility)
        self.__button_slice_z.on_left_mouse_button_clicked = (
            self.__change_slice_z_visibility)
        
        self.__slider_label_intensities = build_label(text='Intensities')
        
        self.__slider_intensities = ui.LineDoubleSlider2D(
            initial_values=slices_loader.intensities_range,
            min_value=slices_loader.data_min, max_value=slices_loader.data_max,
            length=length, line_width=lw, outer_radius=radius, font_size=fs,
            text_template=tt)
        
        color_double_slider(self.__slider_intensities)
        
        self.__slider_intensities.on_change = self.__change_intensity
        
        self.__buttons_label_colormap = build_label(text='Colormap')
        
        self.__supported_colormaps = {
            'Gray': 'gray', 'Bone': 'bone', 'Cividis': 'cividis',
            'Inferno': 'inferno', 'Magma': 'magma', 'Viridis': 'viridis',
            'Jet': 'jet', 'Pastel 1': 'Pastel1', 'Distinguishable': 'dist'}
        
        self.__selected_colormap_idx = 0
        selected_colormap = list(self.__supported_colormaps)[
            self.__selected_colormap_idx]
        self.__colormap = self.__supported_colormaps[selected_colormap]
        
        self.__label_selected_colormap = build_label(text=selected_colormap)
        
        self.__button_previous_colormap = ui.Button2D(
            icon_fnames=[('left', read_viz_icons(fname='circle-left.png'))],
            size=(25, 25))
        
        self.__button_next_colormap = ui.Button2D(
            icon_fnames=[('right', read_viz_icons(fname='circle-right.png'))],
            size=(25, 25))
        
        self.__button_previous_colormap.on_left_mouse_button_clicked = (
            self.__change_colormap_previous)
        self.__button_next_colormap.on_left_mouse_button_clicked = (
            self.__change_colormap_next)
        
        data_ndim = len(self.__data_shape)
        
        if data_ndim == 4:
            self.__slider_label_volume = build_label(text='Volume')
            
            self.__selected_volume_idx = 0
            
            self.__slider_volume = ui.LineSlider2D(
                initial_value=self.__selected_volume_idx,
                max_value=self.__data_shape[-1] - 1, length=length,
                line_width=lw, outer_radius=radius, font_size=fs,
                text_template=tt)
            
            color_single_slider(self.__slider_volume)
            
            self.__slider_volume.handle_events(
                self.__slider_volume.handle.actor)
            self.__slider_volume.on_left_mouse_button_released = (
                self.__change_volume)
    
    def __change_colormap_previous(self, i_ren, _obj, _button):
        selected_colormap_idx = self.__selected_colormap_idx - 1
        if selected_colormap_idx < 0:
            selected_colormap_idx = len(self.__supported_colormaps) - 1
        self.__selected_colormap_idx = selected_colormap_idx
        selected_colormap = list(self.__supported_colormaps)[
            self.__selected_colormap_idx]
        self.__label_selected_colormap.message = selected_colormap
        self.__colormap = self.__supported_colormaps[selected_colormap]
        self.__update_colormap()
        i_ren.force_render()
    
    def __change_colormap_next(self, i_ren, _obj, _button):
        selected_color_idx = self.__selected_colormap_idx + 1
        if selected_color_idx >= len(self.__supported_colormaps):
            selected_color_idx = 0
        self.__selected_colormap_idx = selected_color_idx
        selected_colormap = list(self.__supported_colormaps)[
            self.__selected_colormap_idx]
        self.__label_selected_colormap.message = selected_colormap
        self.__colormap = self.__supported_colormaps[selected_colormap]
        self.__update_colormap()
        i_ren.force_render()
    
    def __change_opacity(self, slider):
        opacity = slider.value
        for slice in self.__actors:
            slice.GetProperty().SetOpacity(opacity)
    
    def __change_intensity(self, slider):
        self.__min_intensity = slider.left_disk_value
        self.__max_intensity = slider.right_disk_value
        self.__update_colormap()
    
    def __change_slice_x(self, slider):
        value = int(np.rint(slider.value))
        self.__actors[0].display_extent(
            value, value, 0, self.__data_shape[1] - 1, 0,
            self.__data_shape[2] - 1)
    
    def __change_slice_y(self, slider):
        value = int(np.rint(slider.value))
        self.__actors[1].display_extent(
            0, self.__data_shape[0] - 1, value, value, 0,
            self.__data_shape[2] - 1)
    
    def __change_slice_z(self, slider):
        value = int(np.rint(slider.value))
        self.__actors[2].display_extent(
            0, self.__data_shape[0] - 1, 0, self.__data_shape[1] - 1, value,
            value)
    
    def __change_slice_x_visibility(self, i_ren, _obj, _button):
        self.__slice_x_visibility = not self.__slice_x_visibility
        self.__slider_slice_x.set_visibility(self.__slice_x_visibility)
        self.__actors[0].SetVisibility(self.__slice_x_visibility)
        _button.next_icon()
        i_ren.force_render()
    
    def __change_slice_y_visibility(self, i_ren, _obj, _button):
        self.__slice_y_visibility = not self.__slice_y_visibility
        self.__slider_slice_y.set_visibility(self.__slice_y_visibility)
        self.__actors[1].SetVisibility(self.__slice_y_visibility)
        _button.next_icon()
        i_ren.force_render()
    
    def __change_slice_z_visibility(self, i_ren, _obj, _button):
        self.__slice_z_visibility = not self.__slice_z_visibility
        self.__slider_slice_z.set_visibility(self.__slice_z_visibility)
        self.__actors[2].SetVisibility(self.__slice_z_visibility)
        _button.next_icon()
        i_ren.force_render()
    
    def __change_volume(self, istyle, obj, slider):
        value = int(np.rint(slider.value))
        
        # TODO: Pass current opacity
        # TODO: Pass current selected slices
        # TODO: Pass visible slices
        """
        loader_data = replace_volume_slice_actors(
            self.__data, self.__global_memory.scene, self.__actors,
            self.__selected_volume_idx, value,
            [self.__min_intensity, self.__max_intensity], affine=self.__affine,
            world_coords=self.__world_coords)
        self.__actors = loader_data[0]
        resliced_shape = loader_data[1]
        data_limits = loader_data[2]
        intensities_range = loader_data[3]
        
        self.__min_intensity = intensities_range[0]
        self.__max_intensity = intensities_range[1]
        self.__update_colormap()
        
        # TODO: Adjust slices sliders
        
        self.__slider_intensities.initial_values = intensities_range
        self.__slider_intensities.min_value = data_limits[0]
        self.__slider_intensities.max_value = data_limits[1]
        """
        
        self.__selected_volume_idx = value
        istyle.force_render()
    
    def __update_colormap(self):
        if self.__colormap == 'dist':
            rgb = colormap.distinguishable_colormap(nb_colors=256)
            rgb = np.asarray(rgb)
        else:
            rgb = colormap.create_colormap(
                np.linspace(self.__min_intensity, self.__max_intensity, 256),
                name=self.__colormap, auto=True)
        num_lut = rgb.shape[0]

        lut = colormap.LookupTable()
        lut.SetNumberOfTableValues(num_lut)
        lut.SetRange(self.__min_intensity, self.__max_intensity)
        for i in range(num_lut):
            r, g, b = rgb[i]
            lut.SetTableValue(i, r, g, b)
        lut.SetRampToLinear()
        lut.Build()
        
        for slice in self.__actors:
            slice.output.SetLookupTable(lut)
            slice.output.Update()
    
    def build(self, tab_id, tab_ui, gmem):
        self.__tab_id = tab_id
        self.__tab_ui = tab_ui
        self.__global_memory = gmem
        
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
            self.__tab_id, self.__button_slice_x, (x_pos, .60))
        self.__tab_ui.add_element(
            self.__tab_id, self.__button_slice_y, (x_pos, .36))
        self.__tab_ui.add_element(
            self.__tab_id, self.__button_slice_z, (x_pos, .13))
        
        x_pos = .12
        
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
        self.__tab_ui.add_element(
            self.__tab_id, self.__buttons_label_colormap, (x_pos, .56))
        
        x_pos = .60
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_intensities, (x_pos, .85))
        self.__tab_ui.add_element(
            self.__tab_id, self.__button_previous_colormap, (x_pos, .54))
        self.__tab_ui.add_element(
            self.__tab_id, self.__label_selected_colormap, (.63, .56))
        self.__tab_ui.add_element(
            self.__tab_id, self.__button_next_colormap, (.73, .54))
        
        data_ndim = len(self.__data_shape)
        
        if data_ndim == 4:
            x_pos = .52
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_label_volume, (x_pos, .38))
            
            x_pos = .60
            self.__tab_ui.add_element(
                self.__tab_id, self.__slider_volume, (x_pos, .38))
    
    @property
    def name(self):
        return self.__name
