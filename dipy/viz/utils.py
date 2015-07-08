
from __future__ import division, print_function, absolute_import

import numpy as np
from dipy.core.ndindex import ndindex
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


def set_input(vtk_object, input):
    """ Generic input for vtk data,
        depending of the type of input and vtk version

    Example
    ----------
    >>> poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)
    """
    if isinstance(input, vtk.vtkPolyData):
        if vtk.VTK_MAJOR_VERSION <= 5:
            vtk_object.SetInput(input)
        else:
            vtk_object.SetInputData(input)
    elif isinstance(input, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(input)

    vtk_object.Update()
    return vtk_object


def evec_from_lines(lines, use_line_dir=True):
    """ Get eigen vectors from lines directions in a 3x3 array

    if use_line_dir is set to False
        only use the points position information

    """

    if use_line_dir:
        lines_dir = []
        for line in lines:
            lines_dir += [line[1:] - line[0:-1]]
        directions = np.vstack(lines_dir)
    else:
        points = np.vstack(lines)
        centered_points = points - np.mean(points, axis=0)
        norm = np.sqrt(np.sum(centered_points**2, axis=1, keepdims=True))
        directions = centered_points/norm

    U, e_val, e_vec = np.linalg.svd(directions, full_matrices=False)
    return e_vec


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


def rescale_to_uint8(data):
    """ Rescales value of a ndarray to 8 bits unsigned integer

    This function rescales the values of the input between 0 and 255,
    then copies it to a new 8 bits unsigned integer array.

    Parameters
    ----------
    data : ndarray

    Return
    ------
    uint8 : ndarray

    Note
    ----
    NANs are clipped to 0. If min equals max, result will be all 0.

    """

    temp = np.array(data, dtype=np.float64)
    temp[np.isnan(temp)] = 0
    temp -= np.min(temp)
    if np.max(temp) != 0.0:
        temp /= np.max(temp)
        temp *= 255.0
    temp = np.array(np.round(temp), dtype=np.uint8)

    return temp
