import warnings

import numpy as np
from scipy import stats

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import actor


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
