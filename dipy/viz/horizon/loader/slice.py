import warnings

import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import actor


def nifti_to_slice_actors(data, affine=None, world_coords=False):
    orig_shape = data.shape
    print('Original shape: {}'.format(orig_shape))
    ndim = data.ndim
    tmp = data
    if ndim == 4:
        if orig_shape[-1] > 3:
            orig_shape = orig_shape[:3]
            # Sometimes, first volume is null, so we try the next one.
            for i in range(orig_shape[-1]):
                tmp = data[..., i]
                value_range = np.percentile(data[..., i], q=[2, 98])
                if np.sum(np.diff(value_range)) != 0:
                    break
        if orig_shape[-1] == 3:
            value_range = (0, 1.)
    if ndim == 3:
        value_range = np.percentile(tmp, q=[2, 98])

    if np.sum(np.diff(value_range)) == 0:
        warnings.warn(
            'Your data does not have any contrast. Please, check the value '
            'range of your data.')

    if not world_coords:
        affine = np.eye(4)

    slice_actor_z = actor.slicer(
        tmp, affine=affine, value_range=value_range, interpolation='nearest',
        picking_tol=0.025)

    tmp_new = slice_actor_z.resliced_array()

    if len(data.shape) == 4:
        if data.shape[-1] == 3:
            print('Resized to RAS shape: {}'.format(tmp_new.shape))
        else:
            print('Resized to RAS shape: {}'.format(
                tmp_new.shape + (data.shape[-1],)))
    else:
        print('Resized to RAS shape: {}'.format(tmp_new.shape))

    shape = tmp_new.shape

    slice_actor_x = slice_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    slice_actor_x.display_extent(
        x_midpoint, x_midpoint, 0, shape[1] - 1, 0, shape[2] - 1)

    slice_actor_y = slice_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    slice_actor_y.display_extent(
        0, shape[0] - 1, y_midpoint, y_midpoint, 0, shape[2] - 1)
    
    return ((slice_actor_x, slice_actor_y, slice_actor_z), tmp, value_range)
