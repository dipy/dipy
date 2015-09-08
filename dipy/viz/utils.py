
from __future__ import division, absolute_import

import os
import numpy as np
from scipy.ndimage import map_coordinates
from nibabel.tmpdirs import InTemporaryDirectory

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
ns, have_numpy_support, _ = optional_package('vtk.util.numpy_support')
_, have_imread, _ = optional_package('Image')
matplotlib, have_mpl, _ = optional_package("matplotlib")

if have_imread:
    from scipy.misc import imread


def vtk_matrix_to_numpy(matrix):
    """ Converts VTK matrix to numpy array.
    """
    if matrix is None:
        return None

    size = (4, 4)
    if isinstance(matrix, vtk.vtkMatrix3x3):
        size = (3, 3)

    mat = np.zeros(size)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i, j] = matrix.GetElement(i, j)

    return mat


def numpy_to_vtk_matrix(array):
    """ Converts a numpy array to a VTK matrix.
    """
    if array is None:
        return None

    if array.shape == (4, 4):
        matrix = vtk.vtkMatrix4x4()
    elif array.shape == (3, 3):
        matrix = vtk.vtkMatrix3x3()
    else:
        raise ValueError("Invalid matrix shape: {0}".format(array.shape))

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            matrix.SetElement(i, j, array[i, j])

    return matrix


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


def get_bounding_box_sizes(actor):
    """ Gets the bounding box sizes of an actor. """
    X1, X2, Y1, Y2, Z1, Z2 = actor.GetBounds()
    return (X2-X1, Y2-Y1, Z2-Z1)


def get_grid_cells_position(shapes, aspect_ratio=16/9., dim=None):
    """ Constructs a XY-grid based on the cells content shape.

    This function generates the coordinates of every grid cell. The width and
    height of every cell correspond to the largest width and the largest height
    respectively. The grid dimensions will automatically be adjusted to respect
    the given aspect ratio unless they are explicitly specified.

    The grid follows a row-major order with the top left corner being at
    coordinates (0,0,0) and the bottom right corner being at coordinates
    (nb_cols*cell_width, -nb_rows*cell_height, 0). Note that the X increases
    while the Y decreases.

    Parameters
    ----------
    shapes : list of tuple of int
        The shape (width, height) of every cell content.
    aspect_ratio : float (optional)
        Aspect ratio of the grid (width/height). Default: 16:9.
    dim : tuple of int (optional)
        Dimension (nb_rows, nb_cols) of the grid, if provided.

    Returns
    -------
    ndarray
        3D coordinates of every grid cell.

    """
    cell_shape = np.r_[np.max(shapes, axis=0), 0]
    cell_aspect_ratio = cell_shape[0]/cell_shape[1]

    count = len(shapes)
    if dim is None:
        # Compute the number of rows and columns.
        n_cols = np.ceil(np.sqrt(count*aspect_ratio / cell_aspect_ratio))
        n_rows = np.ceil(count / n_cols)
        assert n_cols * n_rows >= count
    else:
        n_rows, n_cols = dim

        if n_cols * n_rows < count:
            raise ValueError("Size is too small, it cannot contain at least {} elements.".format(count))

    # Use indexing="xy" so the cells are in row-major (C-order). Also,
    # the Y coordinates are negative so the cells are order from top to bottom.
    X, Y, Z = np.meshgrid(np.arange(n_cols), -np.arange(n_rows), [0], indexing="xy")
    return cell_shape * np.array([X.flatten(), Y.flatten(), Z.flatten()]).T


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


def matplotlib_figure_to_numpy(fig, dpi=100, fname=None, flip_up_down=True,
                               transparent=False):
    r""" Convert a Matplotlib figure to a 3D numpy array with RGBA channels

    Parameters
    ----------
    fig : obj,
        A matplotlib figure object

    dpi : int
        Dots per inch

    fname : str
        If ``fname`` is given then the array will be saved as a png to this
        position.

    flip_up_down : bool
        The origin is different from matlplotlib default and VTK's default
        behaviour (default True).

    transparent : bool
        Make background transparent (default False).

    Returns
    -------
    arr : ndarray
        a numpy 3D array of RGBA values

    Notes
    ------
    The safest way to read the pixel values from the figure was to save them
    using savefig as a png and then read again the png. There is a cleaner
    way found here http://www.icare.univ-lille1.fr/drupal/node/1141 where
    you can actually use fig.canvas.tostring_argb() to get the values directly
    without saving to the disk. However, this was not stable across different
    machines and needed more investigation from what time permited.
    """

    if fname is None:
        with InTemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'tmp.png')
            fig.savefig(fname, dpi=dpi, transparent=transparent,
                        bbox_inches='tight', pad_inches=0)
            arr = imread(fname)
    else:
        fig.savefig(fname, dpi=dpi, transparent=transparent,
                    bbox_inches='tight', pad_inches=0)
        arr = imread(fname)

    if flip_up_down:
        arr = np.flipud(arr)

    return arr
