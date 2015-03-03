
from __future__ import division, print_function, absolute_import

import numpy as np
from dipy.core.ndindex import ndindex

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

#import vtk
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
    if isinstance(input,vtk.vtkPolyData):
        if vtk.VTK_MAJOR_VERSION <= 5:
            vtk_object.SetInput(input)
        else:
            vtk_object.SetInputData(input)
    elif isinstance(input,vtk.vtkAlgorithmOutput):
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
        centered_points = points - np.mean(points,axis=0)
        norm = np.sqrt(np.sum(centered_points**2,axis=1, keepdims=True))
        directions = centered_points/norm

    U, e_val, e_vec = np.linalg.svd(directions, full_matrices=False)
    return e_vec


def rotation_from_lines(lines, use_line_dir=True, use_full_eig=False):
    """ Get the rotation from lines directions in vtk.vtkTransform() object

    if use_line_dir is set to False
        only use the points position information

    if use_full_eig is set to True
        the rotation will be the full eigen_vector matrix
        (not only the Yaw and Pitch)

    Example
    ----------
    >>> camera = renderer.GetActiveCamera()
    >>> rotation = rotation_from_lines(lines)
    >>> camera.ApplyTransform(rotation)
    >>> fvtk.show(renderer)
    """
    e_vec = evec_from_lines(lines, use_line_dir)

    matrix = vtk.vtkMatrix4x4()

    if use_full_eig:
        for (i, j) in ndindex((3, 3)) :
            matrix.SetElement(j, i, e_vec[i,j])

    else:
        v1 = e_vec[2]
        v2 = np.array([0,0,1])
        v3 = np.cross(v1,v2)
        v3 = v3/(np.sqrt(np.sum(v3**2)))
        v4 = np.cross(v3,v1)
        v4 = v4/(np.sqrt(np.sum(v4**2)))

        m1 = np.array([v1,v4,v3])
        cos = np.dot(v2,v1)
        sin = np.dot(v2,v4)
        m2 = np.array([[cos,sin,0],[-sin,cos,0],[0,0,1]])

        m = np.dot( np.dot(m1.T,m2), m1)

        for (i, j) in ndindex((3, 3)) :
            matrix.SetElement(i, j, m[i,j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    return transform


def trilinear_interp(input_array, indices):
    """ Evaluate the input_array data at the given indices
    """

    assert (input_array.ndim > 2 )," array need to be at least 3dimensions"
    assert (input_array.ndim < 5 )," dont support array with more than 4 dims"

    x_indices = indices[:,0]
    y_indices = indices[:,1]
    z_indices = indices[:,2]

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    #Check if xyz1 is beyond array boundary:
    x1[np.where(x1==input_array.shape[0])] = x0.max()
    y1[np.where(y1==input_array.shape[1])] = y0.max()
    z1[np.where(z1==input_array.shape[2])] = z0.max()

    if input_array.ndim == 3:
        x = x_indices - x0
        y = y_indices - y0
        z = z_indices - z0
    elif input_array.ndim == 4:
        x = np.expand_dims(x_indices - x0, axis = 1)
        y = np.expand_dims(y_indices - y0, axis = 1)
        z = np.expand_dims(z_indices - z0, axis = 1)

    output = (input_array[x0,y0,z0]*(1-x)*(1-y)*(1-z) +
                 input_array[x1,y0,z0]*x*(1-y)*(1-z) +
                 input_array[x0,y1,z0]*(1-x)*y*(1-z) +
                 input_array[x0,y0,z1]*(1-x)*(1-y)*z +
                 input_array[x1,y0,z1]*x*(1-y)*z +
                 input_array[x0,y1,z1]*(1-x)*y*z +
                 input_array[x1,y1,z0]*x*y*(1-z) +
                 input_array[x1,y1,z1]*x*y*z)

    return output


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


s