import warnings

import numpy as np
from scipy import stats

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import actor
    from fury.utils import apply_affine


class SlicesVisualizer:
    def __init__(
        self, interactor, scene, data, affine=None, world_coords=False,
        percentiles=[2, 98]):
        
        self.__interactor = interactor
        self.__scene = scene
        self.__data = data
        self.__affine = affine
        
        if not world_coords:
            self.__affine = np.eye(4)
        
        self.__slice_actors = [None] * 3
        
        self.__data_ndim = data.ndim
        self.__data_shape = data.shape
        
        print(f'Original shape: {self.__data_shape}')
        
        vol_data = self.__data
        if self.__data_ndim == 4:
            for i in range(self.__data.shape[-1]):
                vol_data = self.__data[..., i]
                self.__int_range = np.percentile(vol_data, percentiles)
                if np.sum(np.diff(self.__int_range)) != 0:
                    break
                else:
                    if i < data.shape[-1] - 1:
                        warnings.warn(
                            f'Volume N°{i} does not have any contrast. '
                            'Please, check the value ranges of your data. '
                            'Moving to the next volume.')
                    else:
                        _evaluate_intensities_range(self.__int_range)
        else:
            self.__int_range = np.percentile(vol_data, percentiles)
            _evaluate_intensities_range(self.__int_range)
        
        self.__vol_max = np.max(vol_data)
        self.__vol_min = np.min(vol_data)
        
        self.__resliced_vol = None
        
        self.__create_and_resize_actors(vol_data, self.__int_range)
        
        self.__sel_slices = np.rint(
            np.asarray(self.__data_shape[:3]) / 2).astype(int)
        
        self.__add_slice_actors_to_scene(self.__sel_slices)
        
        self.__picker_callback = None
        self.__picked_voxel_actor = None
    
    def __add_slice_actors_to_scene(self, visible_slices):
        self.__slice_actors[0].display_extent(
            visible_slices[0], visible_slices[0], 0, self.__data_shape[1] - 1,
            0, self.__data_shape[2] - 1)
        
        self.__slice_actors[1].display_extent(
            0, self.__data_shape[0] - 1, visible_slices[1], visible_slices[1],
            0, self.__data_shape[2] - 1)
        
        self.__slice_actors[2].display_extent(
            0, self.__data_shape[0] - 1, 0, self.__data_shape[1] - 1,
            visible_slices[2], visible_slices[2])
        
        for act in self.__slice_actors:
            self.__scene.add(act)
    
    def __create_and_resize_actors(self, vol_data, value_range):
        self.__slice_actors[0] = actor.slicer(
            vol_data, affine=self.__affine, value_range=value_range,
            interpolation='nearest')
        
        self.__resliced_vol = self.__slice_actors[0].resliced_array()
        
        self.__slice_actors[1] = self.__slice_actors[0].copy()
        self.__slice_actors[2] = self.__slice_actors[0].copy()
        
        for slice_actor in self.__slice_actors:
            slice_actor.AddObserver(
                'LeftButtonPressEvent', self.__left_click_picker_callback, 1.)
        
        if self.__data_ndim == 4:
            self.__data_shape = (
                self.__resliced_vol.shape + (self.__data.shape[-1],))
        else:
            self.__data_shape = self.__resliced_vol.shape
        print(f'Resized to RAS shape: {self.__data_shape}')
    
    def __left_click_picker_callback(self, obj, event):
        # TODO: Find out why this is not triggered when opacity < 1
        event_pos = self.__interactor.GetEventPosition()

        obj.picker.Pick(event_pos[0], event_pos[1], 0, self.__scene)

        i, j, k = obj.picker.GetPointIJK()
        res = self.__resliced_vol[i, j, k]
        
        try:
            message = f'{res:.2f}'
        except TypeError:
            message = f'{res[0]:.2f} {res[1]:.2f} {res[2]:.2f}'
        message = f'({i}, {j}, {k}) = {message}'
        self.__picker_callback(message)
        # TODO: Fix this
        #self.__replace_picked_voxel_actor(i, j, k)
    
    def __replace_picked_voxel_actor(self, x, y, z):
        if self.__picked_voxel_actor:
            self.__scene.rm(self.__picked_voxel_actor)
        pnt = np.asarray([[x, y, z]])
        pnt = apply_affine(self.__affine, pnt)
        self.__picked_voxel_actor = actor.dot(
            pnt, colors=(.9, .4, .0), dot_size=10)
        self.__scene.add(self.__picked_voxel_actor)
    
    def change_volume(self, prev_idx, next_idx, intensities, visible_slices):
        vol_data = self.__data[..., prev_idx]
        # NOTE: Supported only in latests versions of scipy
        # percs = stats.percentileofscore(np.ravel(vol_data), intensities)
        perc_0 = stats.percentileofscore(np.ravel(vol_data), intensities[0])
        perc_1 = stats.percentileofscore(np.ravel(vol_data), intensities[1])
        vol_data = self.__data[..., next_idx]
        value_range = np.percentile(vol_data, [perc_0, perc_1])
        if np.sum(np.diff(self.__int_range)) == 0:
            return False
        
        self.__int_range = value_range
        
        self.__vol_max = np.max(vol_data)
        self.__vol_min = np.min(vol_data)
        
        for slice_actor in self.__slice_actors:
            self.__scene.rm(slice_actor)
        
        self.__create_and_resize_actors(vol_data, self.__int_range)
        
        self.__add_slice_actors_to_scene(visible_slices)
        
        return True
    
    def register_picker_callback(self, callback):
        self.__picker_callback = callback
    
    @property
    def data_shape(self):
        return self.__data_shape
    
    @property
    def intensities_range(self):
        return self.__int_range
    
    @property
    def selected_slices(self):
        return self.__sel_slices
    
    @property
    def slice_actors(self):
        return self.__slice_actors
    
    @property
    def volume_max(self):
        return self.__vol_max
    
    @property
    def volume_min(self):
        return self.__vol_min


def _evaluate_intensities_range(intensities_range):
    if np.sum(np.diff(intensities_range)) == 0:
        raise ValueError(
            'Your data does not have any contrast. Please, check the '
            'value range of your data.')
