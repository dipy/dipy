import warnings

import numpy as np
from functools import partial

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import (HorizonTab, build_label, build_slider,
                                    build_switcher, build_checkbox)

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import colormap


class SlicesTab(HorizonTab):
    def __init__(self, slices_visualizer, slice_id=0,
                 force_render=lambda _element: None ):
        super().__init__()
        self._visualizer = slices_visualizer

        if slice_id == 0:
            self._name = 'Image'
        else:
            self._name = f'Image {slice_id}'

        self._force_render = force_render
        self._tab_id = 0

        # Providing callback to synchronize slices
        self.on_slice_change = lambda _tab_id, _x, _y, _z: None

        # Opacity slider
        self._slice_opacity_label, self._slice_opacity = build_slider(
            initial_value=1.,
            max_value=1.,
            text_template='{ratio:.0%}',
            on_change=self._change_opacity,
            label='Opacity'
        )

        # Slice X slider
        self._slice_x_label, self._slice_x = build_slider(
            initial_value=self._visualizer.selected_slices[0],
            max_value=self._visualizer.data_shape[0] - 1,
            text_template='{value:.0f}',
            on_moving_slider=self._change_slice_x,
            on_value_changed=self._adjust_slice_x,
            label='X Slice'
        )

        self._change_slice_visibility_x = partial(
            self._update_slice_visibility, selected_slice=self._slice_x,
            actor=self._visualizer.slice_actors[0])

        self._slice_x_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._change_slice_visibility_x)

        # Slice Y slider

        self._slice_y_label, self._slice_y = build_slider(
            initial_value=self._visualizer.selected_slices[1],
            max_value=self._visualizer.data_shape[1] - 1,
            text_template='{value:.0f}',
            on_moving_slider=self._change_slice_y,
            on_value_changed=self._adjust_slice_y,
            label='Y Slice'
        )

        self._change_slice_visibility_y = partial(
            self._update_slice_visibility, selected_slice=self._slice_y,
            actor=self._visualizer.slice_actors[1])

        self._slice_y_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._change_slice_visibility_y)

        # Slice Z slider
        self._slice_z_label, self._slice_z = build_slider(
            initial_value=self._visualizer.selected_slices[2],
            max_value=self._visualizer.data_shape[2] - 1,
            text_template='{value:.0f}',
            on_moving_slider=self._change_slice_z,
            on_value_changed=self._adjust_slice_z,
            label='Z Slice'
        )

        self._change_slice_visibility_z = partial(
            self._update_slice_visibility, selected_slice=self._slice_z,
            actor=self._visualizer.slice_actors[2])

        self._slice_z_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._change_slice_visibility_z)

        # Slider for intensities
        self._intensities_label, self._intensities = build_slider(
            initial_value=self._visualizer.intensities_range,
            min_value=self._visualizer.volume_min,
            max_value=self._visualizer.volume_max,
            text_template='{value:.0f}',
            on_change=self._change_intensity,
            label='Intensities',
            is_double_slider=True
        )

        # Switcher for colormap
        self._supported_colormap = [
            {'label': 'Gray', 'value': 'gray'},
            {'label': 'Bone', 'value': 'bone'},
            {'label': 'Cividis', 'value': 'cividis'},
            {'label': 'Inferno', 'value': 'inferno'},
            {'label': 'Magma', 'value': 'magma'},
            {'label': 'Viridis', 'value': 'viridis'},
            {'label': 'Jet', 'value': 'jet'},
            {'label': 'Pastel 1', 'value': 'Pastel1'},
            {'label': 'Distinct', 'value': 'dist'},

        ]

        self._colormap_switcher_label, self._colormap_switcher = build_switcher(
            items=self._supported_colormap,
            label='Colormap',
            on_value_changed=self._change_color_map
        )

        # Text section for voxel
        self._voxel_label = build_label(text='Voxel', is_horizon_label=True)
        self._voxel_data = build_label(text='', is_horizon_label=True)

        self._visualizer.register_picker_callback(self._change_picked_voxel)

        self.register_elements(
            self._slice_opacity_label,
            self._slice_opacity,
            self._slice_x_toggle,
            self._slice_x_label,
            self._slice_x,
            self._slice_y_toggle,
            self._slice_y_label,
            self._slice_y,
            self._slice_z_toggle,
            self._slice_z_label,
            self._slice_z,
            self._intensities_label,
            self._intensities,
            self._colormap_switcher_label,
            self._colormap_switcher,
            self._voxel_label,
            self._voxel_data
        )

        # Slider for volume if data provided have volume information
        if len(self._visualizer.data_shape) == 4:
            self._volume_label, self._volume = build_slider(
                initial_value=0,
                max_value=self._visualizer.data_shape[-1] - 1,
                on_moving_slider=self._change_volume,
                text_template='{value:.0f}',
                label='Volume'
            )
            self.register_elements(self._volume_label, self._volume)

    def _change_color_map(self, _idx, _value):
        self._update_colormap()
        self._force_render(self)

    def _change_intensity(self, slider):
        self._intensities.selected_value[0] = slider.left_disk_value
        self._intensities.selected_value[1] = slider.right_disk_value
        self._update_colormap()

    def _change_opacity(self, slider):
        self._slice_opacity.selected_value = slider.value
        self._update_opacities()

    def _change_picked_voxel(self, message):
        self._voxel_data.obj.message = message
        self._voxel_data.selected_value = message

    # Slice change callbacks
    def _change_slice_x(self, slider, synchronized_value=None):
        self._change_slice_value(self._slice_x, slider, synchronized_value)
        self._visualizer.slice_actors[0].display_extent(
            self._slice_x.selected_value, self._slice_x.selected_value,
            0, self._visualizer.data_shape[1] - 1, 0,
            self._visualizer.data_shape[2] - 1)

    def _adjust_slice_x(self, slider):
        self._change_slice_x(slider, slider.value)

    def _change_slice_y(self, slider, synchronized_value=None):
        self._change_slice_value(self._slice_y, slider, synchronized_value)
        self._visualizer.slice_actors[1].display_extent(
            0, self._visualizer.data_shape[0] - 1, self._slice_y.selected_value,
            self._slice_y.selected_value, 0, self._visualizer.data_shape[2] - 1)

    def _adjust_slice_y(self, slider):
        self._change_slice_y(slider, slider.value)

    def _change_slice_z(self, slider, synchronized_value=None):
        self._change_slice_value(self._slice_z, slider, synchronized_value)
        self._visualizer.slice_actors[2].display_extent(
            0, self._visualizer.data_shape[0] - 1, 0,
            self._visualizer.data_shape[1] - 1,
            self._slice_z.selected_value, self._slice_z.selected_value)

    def _adjust_slice_z(self, slider):
        self._change_slice_z(slider, slider.value)

    def _change_slice_value(self, selected_slice, slider, new_value=None):
        if new_value:
            selected_slice.selected_value = int(np.rint(new_value))
        else:
            selected_slice.selected_value = int(np.rint(slider.value))
            self.on_slice_change(
                self._tab_id,
                self._slice_x.selected_value,
                self._slice_y.selected_value,
                self._slice_z.selected_value
            )

    def _update_slice_visibility(
            self, checkboxes, selected_slice, actor, visibility=None):

        if checkboxes is not None and '' in checkboxes.checked_labels:
            visibility = True
        elif visibility is None:
            visibility = False

        selected_slice.visibility = visibility
        selected_slice.obj.set_visibility(visibility)
        actor.SetVisibility(visibility)


    def _change_volume(self, slider):
        value = int(np.rint(slider.value))
        if value != self._volume.selected_value:
            visible_slices = (
                self._slice_x.obj.value, self._slice_y.selected_value,
                self._slice_z.selected_value)
            valid_vol = self._visualizer.change_volume(
                self._volume.selected_value, value,
                [self._intensities.selected_value[0],
                 self._intensities.selected_value[1]], visible_slices)
            if not valid_vol:
                warnings.warn(
                    f'Volume N°{value} does not have any contrast. Please, '
                    'check the value ranges of your data. Returning to '
                    'previous volume.')
                self._volume.obj.value = self._volume.selected_value
            else:
                intensities_range = self._visualizer.intensities_range

                # Updating the colormap
                self._intensities.selected_value[0] = intensities_range[0]
                self._intensities.selected_value[1] = intensities_range[1]
                self._update_colormap()

                # Updating intensities slider
                self._intensities.obj.initial_values = intensities_range
                self._intensities.obj.min_value = (
                    self._visualizer.volume_min)
                self._intensities.obj.max_value = (
                    self._visualizer.volume_max)
                self._intensities.obj.update(0)
                self._intensities.obj.update(1)

                # Updating opacities
                self._update_opacities()

                # Updating visibilities
                slices = [self._slice_x, self._slice_y, self._slice_z]
                for s, i in enumerate(slices):
                    self._update_slice_visibility(
                        None,
                        s.obj,
                        self._visualizer.slice_actors[i],
                        s.visibility
                    )

                self._volume.selected_value = value
                self._force_render(self)

    def _update_colormap(self):
        if self._colormap_switcher.selected_value[1] == 'dist':
            rgb = colormap.distinguishable_colormap(nb_colors=256)
            rgb = np.asarray(rgb)
        else:
            rgb = colormap.create_colormap(
                np.linspace(self._intensities.selected_value[0],
                            self._intensities.selected_value[1], 256),
                name=self._colormap_switcher.selected_value[1], auto=True)
        num_lut = rgb.shape[0]

        lut = colormap.LookupTable()
        lut.SetNumberOfTableValues(num_lut)
        lut.SetRange(self._intensities.selected_value[0],
                     self._intensities.selected_value[1])
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
            slice_actor.GetProperty().SetOpacity(
                self._slice_opacity.selected_value)

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
        if not self._slice_x.obj.value == x_slice:
            self._slice_x.obj.value = x_slice

        if not self._slice_y.obj.value == y_slice:
            self._slice_y.obj.value = y_slice

        if not self._slice_z.obj.value == z_slice:
            self._slice_z.obj.value = z_slice

    def build(self, tab_id, _tab_ui):
        self._tab_id = tab_id

        x_pos = .02
        self._slice_x_toggle.position = (x_pos, .62)
        self._slice_y_toggle.position = (x_pos, .38)
        self._slice_z_toggle.position = (x_pos, .15)

        x_pos = .05
        self._slice_opacity_label.position = (x_pos, .85)
        self._slice_x_label.position = (x_pos, .62)
        self._slice_y_label.position = (x_pos, .38)
        self._slice_z_label.position = (x_pos, .15)

        x_pos = .10
        self._slice_opacity.position = (x_pos, .85)
        self._slice_x.position = (x_pos, .62)
        self._slice_y.position = (x_pos, .38)
        self._slice_z.position = (x_pos, .15)

        x_pos = .52
        self._intensities_label.position = (x_pos, .85)
        self._colormap_switcher_label.position = (x_pos, .56)
        self._voxel_label.position = (x_pos, .38)

        x_pos = .60
        self._intensities.position = (x_pos, .85)
        self._colormap_switcher.position = [
            (x_pos, .54), (0.63, .54), (0.69, .54)
        ]
        self._voxel_data.position = (x_pos, .38)

        if len(self._visualizer.data_shape) == 4:
            x_pos = .52
            self._volume_label.position(x_pos, .15)

            x_pos = .60
            self._volume.position = (x_pos, .15)

    @property
    def name(self):
        return self._name

    @property
    def tab_id(self):
        """
        Id of the tab.
        """
        return self._tab_id

    @property
    def tab_type(self):
        return 'slices_tab'
