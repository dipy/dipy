
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.ndimage import map_coordinates

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
ns, have_numpy_support, _ = optional_package('vtk.util.numpy_support')


def numpy_to_vtk_points(points):
    """ numpy points array to a vtk points array
    """
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True))
    return vtk_points


def numpy_to_vtk_colors(colors):
    """ numpy color array to a vtk color array

    if colors are not already in UNSIGNED_CHAR
        you may need to multiply by 255.

    Example
    ----------
    >>>  vtk_colors = numpy_to_vtk_colors(255 * float_array)
    """
    vtk_colors = ns.numpy_to_vtk(np.asarray(colors), deep=True,
                                 array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors


def set_input(vtk_object, inp):
    """ Generic input function which takes into account VTK 5 or 6

    Parameters
    ----------
    vtk_object: vtk object
    inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput

    Returns
    -------
    vtk_object

    Example
    ----------
    >>> poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)
    """
    if isinstance(inp, vtk.vtkPolyData) \
       or isinstance(inp, vtk.vtkImageData):
        if vtk.VTK_MAJOR_VERSION <= 5:
            vtk_object.SetInput(inp)
        else:
            vtk_object.SetInputData(inp)
    elif isinstance(inp, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(inp)

    vtk_object.Update()
    return vtk_object


def map_coordinates_3d_4d(input_array, indices):
    """ Evaluate the input_array data at the given indices
    using trilinear interpolation

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array
    """

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=1)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=1)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def auto_camera(actor, zoom=10, relative='max'):
    """ Automatically calculate the position of the camera given an actor

    """

    bounds = actor.GetBounds()

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    bounds = np.array(bounds).reshape(3, 2)
    center_bb = bounds.mean(axis=1)
    widths_bb = np.abs(bounds[:, 0] - bounds[:, 1])

    corners = np.array([[x_min, y_min, z_min],
                        [x_min, y_min, z_max],
                        [x_min, y_max, z_min],
                        [x_min, y_max, z_max],
                        [x_max, y_min, z_min],
                        [x_max, y_min, z_max],
                        [x_max, y_max, z_min],
                        [x_max, y_max, z_max]])

    x_plane_min = np.array([[x_min, y_min, z_min],
                            [x_min, y_min, z_max],
                            [x_min, y_max, z_min],
                            [x_min, y_max, z_max]])

    x_plane_max = np.array([[x_max, y_min, z_min],
                            [x_max, y_min, z_max],
                            [x_max, y_max, z_min],
                            [x_max, y_max, z_max]])

    y_plane_min = np.array([[x_min, y_min, z_min],
                            [x_min, y_min, z_max],
                            [x_max, y_min, z_min],
                            [x_max, y_min, z_max]])

    y_plane_max = np.array([[x_min, y_max, z_min],
                            [x_min, y_max, z_max],
                            [x_max, y_max, z_min],
                            [x_max, y_max, z_max]])

    z_plane_min = np.array([[x_min, y_min, z_min],
                            [x_min, y_max, z_min],
                            [x_max, y_min, z_min],
                            [x_max, y_max, z_min]])

    z_plane_max = np.array([[x_min, y_min, z_max],
                            [x_min, y_max, z_max],
                            [x_max, y_min, z_max],
                            [x_max, y_max, z_max]])

    which_plane = np.argmin(widths_bb)

    if which_plane == 0:
        if relative == 'max':
            plane = x_plane_max
        else:
            plane = x_plane_min
        view_up = np.array([0, 1, 0])

    if which_plane == 1:
        if relative == 'max':
            plane = y_plane_max
        else:
            plane = y_plane_min
        view_up = np.array([0, 0, 1])

    if which_plane == 2:
        if relative == 'max':
            plane = z_plane_max
        else:
            plane = z_plane_min
        view_up = np.array([0, 1, 0])

    initial_position = np.mean(plane, axis=0)

    position = center_bb + zoom * (initial_position - center_bb)

    return position, center_bb, view_up, corners, plane