import warnings
from logging import warning

import numpy as np
from scipy import stats

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import actor


class SlicesLoader:
    def __init__(
        self, scene, data, affine=None, world_coords=False,
        percentiles=[2, 98]):
        
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
                        evaluate_intensities_range(self.__int_range)
        else:
            self.__int_range = np.percentile(vol_data, percentiles)
            evaluate_intensities_range(self.__int_range)
        
        self.__vol_max = np.max(vol_data)
        self.__vol_min = np.min(vol_data)
        
        self.__create_and_resize_actors(vol_data, self.__int_range)
        
        visible_slices = np.rint(
            np.asarray(self.__data_shape[:3]) / 2).astype(int)
        
        self.__add_slice_actors_to_scene(visible_slices)
    
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
        
        resliced_vol = self.__slice_actors[0].resliced_array()
        
        self.__slice_actors[1] = self.__slice_actors[0].copy()
        self.__slice_actors[2] = self.__slice_actors[0].copy()
        
        if self.__data_ndim == 4:
            self.__data_shape = resliced_vol.shape + (self.__data.shape[-1],)
        else:
            self.__data_shape = resliced_vol.shape
        print(f'Resized to RAS shape: {self.__data_shape}')
    
    @property
    def data_shape(self):
        return self.__data_shape
    
    @property
    def intensities_range(self):
        return self.__int_range
    
    @property
    def slice_actors(self):
        return self.__slice_actors
    
    @property
    def volume_max(self):
        return self.__vol_max
    
    @property
    def volume_min(self):
        return self.__vol_min


def evaluate_intensities_range(intensities_range):
    if np.sum(np.diff(intensities_range)) == 0:
        raise ValueError(
            'Your data does not have any contrast. Please, check the '
            'value range of your data.')


def add_slice_actors(
    data, scene, affine=None, world_coords=False, percentiles=[2, 98]):
    orig_shape = data.shape
    print(f'Original shape: {orig_shape}')
    
    ndim = data.ndim
    tmp_data = data
    if ndim == 4:
        tmp_data = data[..., 0]
    
    value_range = np.percentile(tmp_data, percentiles)

    if np.sum(np.diff(value_range)) == 0:
        warnings.warn(
            'Your data does not have any contrast. Please, check the value '
            'range of your data.')

    if not world_coords:
        affine = np.eye(4)

    slice_actor_z = actor.slicer(
        tmp_data, affine=affine, value_range=value_range,
        interpolation='nearest')

    tmp_new = slice_actor_z.resliced_array()

    if ndim == 4:
        shape = tmp_new.shape + (data.shape[-1],)
    else:
        shape = tmp_new.shape
    print(f'Resized to RAS shape: {shape}')

    slice_actor_x = slice_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    slice_actor_x.display_extent(
        x_midpoint, x_midpoint, 0, shape[1] - 1, 0, shape[2] - 1)

    slice_actor_y = slice_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    slice_actor_y.display_extent(
        0, shape[0] - 1, y_midpoint, y_midpoint, 0, shape[2] - 1)
    
    scene.add(slice_actor_x)
    scene.add(slice_actor_y)
    scene.add(slice_actor_z)
    
    return ((slice_actor_x, slice_actor_y, slice_actor_z), shape,
            (tmp_data.min(), tmp_data.max()), value_range)


def replace_volume_slice_actors(
    data, scene, actors, prev_idx, new_idx, intensities, affine=None,
    world_coords=False):
    for act in actors:
        scene.rm(act)
        
    tmp_data = np.ravel(data[..., prev_idx])
    percentiles = stats.percentileofscore(tmp_data, intensities)
    
    tmp_data = data[..., new_idx]
    value_range = np.percentile(tmp_data, percentiles)
    
    if np.sum(np.diff(value_range)) == 0:
        warnings.warn(
            'This volume does not have any contrast. Please, check the value '
            'range of your data.')
    
    if not world_coords:
        affine = np.eye(4)

    slice_actor_z = actor.slicer(
        tmp_data, affine=affine, value_range=value_range,
        interpolation='nearest')
    
    tmp_new = slice_actor_z.resliced_array()
    
    ndim = data.ndim

    if ndim == 4:
        shape = tmp_new.shape + (data.shape[-1],)
    else:
        shape = tmp_new.shape
    
    slice_actor_x = slice_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    slice_actor_x.display_extent(
        x_midpoint, x_midpoint, 0, shape[1] - 1, 0, shape[2] - 1)

    slice_actor_y = slice_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    slice_actor_y.display_extent(
        0, shape[0] - 1, y_midpoint, y_midpoint, 0, shape[2] - 1)
    
    scene.add(slice_actor_x)
    scene.add(slice_actor_y)
    scene.add(slice_actor_z)
    
    return ((slice_actor_x, slice_actor_y, slice_actor_z), shape,
            (tmp_data.min(), tmp_data.max()), value_range)
