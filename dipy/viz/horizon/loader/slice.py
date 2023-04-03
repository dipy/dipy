import warnings

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.gmem import GlobalHorizon

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import actor


def add_slice_actors(data, scene, affine=None, world_coords=False):
    orig_shape = data.shape
    print(f'Original shape: {orig_shape}')
    ndim = data.ndim
    tmp_data = data
    if ndim == 4:
        tmp_data = data[..., 0]
    elif 3 > ndim > 4:
        raise ValueError(f'Unsupported data dimension: {ndim}')
    print(f'{tmp_data.min()} - {tmp_data.max()}')
    
    value_range = np.percentile(tmp_data, [2, 98])

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
    print(f'{tmp_new.min()} - {tmp_new.max()}')

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
            (data.min(), data.max()), value_range)


def replace_slice_actors(data, scene, actors, affine=None, world_coords=False):
    pass
