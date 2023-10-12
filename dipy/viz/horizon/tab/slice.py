import warnings

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import (HorizonTab, build_label, build_slider,
                                    build_checkbox, SLICES_TAB)

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import colormap, ui
    from fury.data import read_viz_icons


class SlicesTab(HorizonTab):
    def __init__(self, slices_visualizer, slice_id=0,
                 force_render=lambda _element: None ):
        super().__init__(SLICES_TAB)
        self._visualizer = slices_visualizer

        if slice_id == 0:
            self._name = 'Image'
        else:
            self._name = f'Image {slice_id}'

        self._force_render = force_render
        self._tab_id = 0
        self._tab_ui = None

        # Providing callback to synchronize slices
        self.on_slice_change = lambda _tab_id, _x, _y, _z: None

        # Opacity slider
        self._slice_opacity = build_slider(
            initial_value=1.,
            max_value=1.,
            text_template='{ratio:.0%}',
            on_change=self._change_opacity,
            label='Opacity'
        )

        # Slice X slider
        self._slice_x = build_slider(
            initial_value=self._visualizer.selected_slices[0],
            max_value=self._visualizer.data_shape[0] - 1,
            text_template='{value:.0f}',
            on_moving_slider=self._change_slice_x,
            on_value_changed=self._adjust_slice_x,
            label='X Slice'
        )
        self._slice_x_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._change_slice_visibility_x)

        # Slice Y slider
        self._slice_y = build_slider(
            initial_value=self._visualizer.selected_slices[1],
            max_value=self._visualizer.data_shape[1] - 1,
            text_template='{value:.0f}',
            on_moving_slider=self._change_slice_y,
            on_value_changed=self._adjust_slice_y,
            label='Y Slice'
        )
        self._slice_y_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._change_slice_visibility_y)

        # Slice Z slider
        self._slice_z = build_slider(
            initial_value=self._visualizer.selected_slices[2],
            max_value=self._visualizer.data_shape[2] - 1,
            text_template='{value:.0f}',
            on_moving_slider=self._change_slice_z,
            on_value_changed=self._adjust_slice_z,
            label='Z Slice'
        )
        self._slice_z_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._change_slice_visibility_z)


        # Slider for intensities
        self._intensities = build_slider(
            initial_value=self._visualizer.intensities_range,
            min_value=self._visualizer.volume_min,
            max_value=self._visualizer.volume_max,
            text_template='{value:.0f}',
            on_change=self._change_intensity,
            label='Intensities',
            is_double_slider=True
        )

        self.__buttons_label_colormap = build_label(text='Colormap')

        self.__supported_colormaps = {
            'Gray': 'gray', 'Bone': 'bone', 'Cividis': 'cividis',
            'Inferno': 'inferno', 'Magma': 'magma', 'Viridis': 'viridis',
            'Jet': 'jet', 'Pastel 1': 'Pastel1', 'Distinct': 'dist'}

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

        data_ndim = len(self._visualizer.data_shape)

        self.__picker_label_voxel = build_label(text='Voxel')

        self.__label_picked_voxel = build_label(text='')

        self._visualizer.register_picker_callback(self.__change_picked_voxel)

        if data_ndim == 4:
            self._volume = build_slider(
                initial_value=0,
                max_value=self._visualizer.data_shape[-1] - 1,
                on_moving_slider=self._change_volume,
                text_template='{value:.0f}',
                label='Volume'
            )


    def __change_colormap_next(self, _i_ren, _obj, _button):
        selected_color_idx = self.__selected_colormap_idx + 1
        if selected_color_idx >= len(self.__supported_colormaps):
            selected_color_idx = 0
        self.__selected_colormap_idx = selected_color_idx
        selected_colormap = list(self.__supported_colormaps)[
            self.__selected_colormap_idx]
        self.__label_selected_colormap.message = selected_colormap
        self.__colormap = self.__supported_colormaps[selected_colormap]
        self._update_colormap()
        self._force_render(self)

    def __change_colormap_previous(self, _i_ren, _obj, _button):
        selected_colormap_idx = self.__selected_colormap_idx - 1
        if selected_colormap_idx < 0:
            selected_colormap_idx = len(self.__supported_colormaps) - 1
        self.__selected_colormap_idx = selected_colormap_idx
        selected_colormap = list(self.__supported_colormaps)[
            self.__selected_colormap_idx]
        self.__label_selected_colormap.message = selected_colormap
        self.__colormap = self.__supported_colormaps[selected_colormap]
        self._update_colormap()
        self._force_render(self)

    def _change_intensity(self, slider):
        self._intensities.slider.selected_value[0] = slider.left_disk_value
        self._intensities.slider.selected_value[1] = slider.right_disk_value
        self._update_colormap()

    def _change_opacity(self, slider):
        self._slice_opacity.slider.selected_value = slider.value
        self._update_opacities()

    def __change_picked_voxel(self, message):
        self.__label_picked_voxel.message = message

    # Slice change callbacks
    def _change_slice_x(self, slider, synchronized_value=None):
        self._change_slice_value(self._slice_x, slider, synchronized_value)
        self._visualizer.slice_actors[0].display_extent(
            self._slice_x.slider.selected_value, self._slice_x.slider.selected_value,
            0, self._visualizer.data_shape[1] - 1, 0, self._visualizer.data_shape[2] - 1)

    def _adjust_slice_x(self, slider):
        self._change_slice_x(slider, slider.value)

    def _change_slice_y(self, slider, synchronized_value=None):
        self._change_slice_value(self._slice_y, slider, synchronized_value)
        self._visualizer.slice_actors[1].display_extent(
            0, self._visualizer.data_shape[0] - 1, self._slice_y.slider.selected_value,
            self._slice_y.slider.selected_value, 0, self._visualizer.data_shape[2] - 1)

    def _adjust_slice_y(self, slider):
        self._change_slice_y(slider, slider.value)

    def _change_slice_z(self, slider, synchronized_value=None):
        self._change_slice_value(self._slice_z, slider, synchronized_value)
        self._visualizer.slice_actors[2].display_extent(
            0, self._visualizer.data_shape[0] - 1, 0, self._visualizer.data_shape[1] - 1,
            self._slice_z.slider.selected_value, self._slice_z.slider.selected_value)

    def _adjust_slice_z(self, slider):
        self._change_slice_z(slider, slider.value)

    def _change_slice_value(self, selected_slice, slider, new_value=None):
        if new_value:
            selected_slice.slider.selected_value = int(np.rint(new_value))
        else:
            selected_slice.slider.selected_value = int(np.rint(slider.value))
            self.on_slice_change(
                self._tab_id,
                self._slice_x.slider.selected_value,
                self._slice_y.slider.selected_value,
                self._slice_z.slider.selected_value
            )

    def _change_slice_visibility_x(self, _checkboxes):
        self._update_slice_visibility(
            self._slice_x,
            self._visualizer.slice_actors[0],
            not self._slice_x.slider.visibility
        )

    def _change_slice_visibility_y(self, _checkboxes):
        self._update_slice_visibility(
            self._slice_y,
            self._visualizer.slice_actors[1],
            not self._slice_y.slider.visibility
        )


    def _change_slice_visibility_z(self, _checkboxes):
        self._update_slice_visibility(
            self._slice_z,
            self._visualizer.slice_actors[2],
            not self._slice_z.slider.visibility
        )

    def _update_slice_visibility(self, selected_slice, actor, visibility):
        selected_slice.slider.visibility = visibility
        selected_slice.slider.obj.set_visibility(visibility)
        actor.SetVisibility(visibility)


    def _change_volume(self, slider):
        value = int(np.rint(slider.value))
        if value != self._volume.slider.selected_value:
            visible_slices = (
                self._slice_x.slider.obj.value, self._slice_y.slider.selected_value,
                self._slice_z.slider.selected_value)
            valid_vol = self._visualizer.change_volume(
                self._volume.slider.selected_value, value,
                [self._intensities.slider.selected_value[0],
                 self._intensities.slider.selected_value[1]], visible_slices)
            if not valid_vol:
                warnings.warn(
                    f'Volume N°{value} does not have any contrast. Please, '
                    'check the value ranges of your data. Returning to '
                    'previous volume.')
                self._volume.slider.obj.value = self._volume.slider.selected_value
            else:
                intensities_range = self._visualizer.intensities_range

                # Updating the colormap
                self._intensities.slider.selected_value[0] = intensities_range[0]
                self._intensities.slider.selected_value[1] = intensities_range[1]
                self._update_colormap()

                # Updating intensities slider
                self._intensities.slider.obj.initial_values = intensities_range
                self._intensities.slider.obj.min_value = (
                    self._visualizer.volume_min)
                self._intensities.slider.obj.max_value = (
                    self._visualizer.volume_max)
                self._intensities.slider.obj.update(0)
                self._intensities.slider.obj.update(1)

                # Updating opacities
                self._update_opacities()

                # Updating visibilities
                slices = [self._slice_x, self._slice_y, self._slice_z]
                for s, i in enumerate(slices):
                    self._update_slice_visibility(
                        s.slider.obj,
                        self._visualizer.actors[i],
                        s.slider.obj
                    )

                self._volume.slider.selected_value = value
                self._force_render(self)

    def _update_colormap(self):
        if self.__colormap == 'dist':
            rgb = colormap.distinguishable_colormap(nb_colors=256)
            rgb = np.asarray(rgb)
        else:
            rgb = colormap.create_colormap(
                np.linspace(self._intensities.slider.selected_value[0],
                            self._intensities.slider.selected_value[1], 256),
                name=self.__colormap, auto=True)
        num_lut = rgb.shape[0]

        lut = colormap.LookupTable()
        lut.SetNumberOfTableValues(num_lut)
        lut.SetRange(self._intensities.slider.selected_value[0],
                     self._intensities.slider.selected_value[1])
        for i in range(num_lut):
            r, g, b = rgb[i]
            lut.SetTableValue(i, r, g, b)
        lut.SetRampToLinear()
        lut.Build()

        for slice_actor in self._visualizer.slice_actors:
            slice_actor.output.SetLookupTable(lut)
            slice_actor.output.Update()

    def _update_opacities(self):
        for slice_actor in self._visualizer.slice_actors:
            slice_actor.GetProperty().SetOpacity(self._slice_opacity.slider.selected_value)

    def update_slices(self, x_slice, y_slice, z_slice):
        """
        Updates slicer positions

        Parameters
        ----------
        x_slice: float
            x-value where the slicer should be placed
        y_slice: float
            y-value where the slicer should be placed
        z_slice: float
            z-value where the slicer should be placed
        """
        if not self._slice_x.slider.obj.value == x_slice:
            self._slice_x.slider.obj.value = x_slice

        if not self._slice_y.slider.obj.value == y_slice:
            self._slice_y.slider.obj.value = y_slice

        if not self._slice_z.slider.obj.value == z_slice:
            self._slice_z.slider.obj.value = z_slice

    def build(self, tab_id, tab_ui, tab_manager):
        self._tab_id = tab_id
        self._tab_ui = tab_ui
        SlicesTab.tab_manager = tab_manager

        x_pos = .02

        self._tab_ui.add_element(
            self._tab_id, self._slice_x_toggle.obj, (x_pos, .62))
        self._tab_ui.add_element(
            self._tab_id, self._slice_y_toggle.obj, (x_pos, .38))
        self._tab_ui.add_element(
            self._tab_id, self._slice_z_toggle.obj, (x_pos, .15))

        x_pos = .05

        self._tab_ui.add_element(
            self._tab_id, self._slice_opacity.label.obj, (x_pos, .85))
        self._tab_ui.add_element(
            self._tab_id, self._slice_x.label.obj, (x_pos, .62))
        self._tab_ui.add_element(
            self._tab_id, self._slice_y.label.obj, (x_pos, .38))
        self._tab_ui.add_element(
            self._tab_id, self._slice_z.label.obj, (x_pos, .15))

        x_pos = .10

        self._tab_ui.add_element(
            self._tab_id, self._slice_opacity.slider.obj, (x_pos, .85))
        self._tab_ui.add_element(
            self._tab_id, self._slice_x.slider.obj, (x_pos, .62))
        self._tab_ui.add_element(
            self._tab_id, self._slice_y.slider.obj, (x_pos, .38))
        self._tab_ui.add_element(
            self._tab_id, self._slice_z.slider.obj, (x_pos, .15))

        x_pos = .52

        self._tab_ui.add_element(
            self._tab_id, self._intensities.label.obj, (x_pos, .85))
        self._tab_ui.add_element(
            self._tab_id, self.__buttons_label_colormap, (x_pos, .56))
        self._tab_ui.add_element(
            self._tab_id, self.__picker_label_voxel, (x_pos, .38))

        x_pos = .60

        self._tab_ui.add_element(
            self._tab_id, self._intensities.slider.obj, (x_pos, .85))
        self._tab_ui.add_element(
            self._tab_id, self.__button_previous_colormap, (x_pos, .54))
        self._tab_ui.add_element(
            self._tab_id, self.__label_selected_colormap, (.63, .56))
        self._tab_ui.add_element(
            self._tab_id, self.__button_next_colormap, (.69, .54))
        self._tab_ui.add_element(
            self._tab_id, self.__label_picked_voxel, (x_pos, .38))

        data_ndim = len(self._visualizer.data_shape)

        if data_ndim == 4:
            x_pos = .52
            self._tab_ui.add_element(
                self._tab_id, self._volume.label.obj, (x_pos, .15))

            x_pos = .60
            self._tab_ui.add_element(
                self._tab_id, self._volume.slider.obj, (x_pos, .15))

    @property
    def name(self):
        return self._name

    @property
    def tab_id(self):
        return self._tab_id
