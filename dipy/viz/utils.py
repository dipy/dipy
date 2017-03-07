
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.ndimage import map_coordinates
from dipy.viz.colormap import line_colors

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
ns, have_numpy_support, _ = optional_package('vtk.util.numpy_support')


def set_input(vtk_object, inp):
    """ Generic input function which takes into account VTK 5 or 6

    Parameters
    ----------
    vtk_object: vtk object
    inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput

    Returns
    -------
    vtk_object

    Notes
    -------
    This can be used in the following way::
        from dipy.viz.utils import set_input
        poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)
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


def numpy_to_vtk_points(points):
    """ Numpy points array to a vtk points array

    Parameters
    ----------
    points : ndarray

    Returns
    -------
    vtk_points : vtkPoints()
    """
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True))
    return vtk_points


def numpy_to_vtk_colors(colors):
    """ Numpy color array to a vtk color array

    Parameters
    ----------
    colors: ndarray

    Returns
    -------
    vtk_colors : vtkDataArray

    Notes
    -----
    If colors are not already in UNSIGNED_CHAR you may need to multiply by 255.

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.viz.utils import numpy_to_vtk_colors
    >>> rgb_array = np.random.rand(100, 3)
    >>> vtk_colors = numpy_to_vtk_colors(255 * rgb_array)
    """
    vtk_colors = ns.numpy_to_vtk(np.asarray(colors), deep=True,
                                 array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors


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


def lines_to_vtk_polydata(lines, colors=None):
    """ Create a vtkPolyData with lines and colors

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), list of arrays, tuple (3,), array (K,), None
        If None then a standard orientation colormap is used for every line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K, 3) is given, where K is the number of points of all
        lines then every point is colored with a different RGB color.
        If an array (K,) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L,) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    Returns
    -------
    poly_data : vtkPolyData
    is_colormap : bool, true if the input color array was a colormap
    """

    # Get the 3d points_array
    points_array = np.vstack(lines)

    nb_lines = len(lines)
    nb_points = len(points_array)

    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    lines_array = []
    # Using np.intp (instead of int64), because of a bug in numpy:
    # https://github.com/nipy/dipy/pull/789
    # https://github.com/numpy/numpy/issues/4384
    points_per_line = np.zeros([nb_lines], np.intp)
    current_position = 0
    for i in lines_range:
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position, end_position)
        current_position = end_position

    lines_array = np.array(lines_array)

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines.GetData().DeepCopy(ns.numpy_to_vtk(lines_array))
    vtk_lines.SetNumberOfCells(nb_lines)

    is_colormap = False
    # Get colors_array (reformat to have colors for each points)
    #           - if/else tested and work in normal simple case
    if colors is None:  # set automatic rgb colors
        cols_arr = line_colors(lines)
        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
        vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
    else:
        cols_arr = np.asarray(colors)
        if cols_arr.dtype == np.object:  # colors is a list of colors
            vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
        else:
            if len(cols_arr) == nb_points:
                if cols_arr.ndim == 1:  # values for every point
                    vtk_colors = ns.numpy_to_vtk(cols_arr, deep=True)
                    is_colormap = True
                elif cols_arr.ndim == 2:  # map color to each point
                    vtk_colors = numpy_to_vtk_colors(255 * cols_arr)

            elif cols_arr.ndim == 1:
                if len(cols_arr) == nb_lines:  # values for every streamline
                    cols_arrx = []
                    for (i, value) in enumerate(colors):
                        cols_arrx += lines[i].shape[0]*[value]
                    cols_arrx = np.array(cols_arrx)
                    vtk_colors = ns.numpy_to_vtk(cols_arrx, deep=True)
                    is_colormap = True
                else:  # the same colors for all points
                    vtk_colors = numpy_to_vtk_colors(
                        np.tile(255 * cols_arr, (nb_points, 1)))

            elif cols_arr.ndim == 2:  # map color to each line
                colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
            else:  # colormap
                #  get colors for each vertex
                cols_arr = map_coordinates_3d_4d(cols_arr, points_array)
                vtk_colors = ns.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

    vtk_colors.SetName("Colors")

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)
    poly_data.GetPointData().SetScalars(vtk_colors)
    return poly_data, is_colormap


def get_polydata_lines(line_polydata):
    """ vtk polydata to a list of lines ndarrays

    Parameters
    ----------
    line_polydata : vtkPolyData

    Returns
    -------
    lines : list
        List of N curves represented as 2D ndarrays
    """
    lines_vertices = ns.vtk_to_numpy(line_polydata.GetPoints().GetData())
    lines_idx = ns.vtk_to_numpy(line_polydata.GetLines().GetData())

    lines = []
    current_idx = 0
    while current_idx < len(lines_idx):
        line_len = lines_idx[current_idx]

        next_idx = current_idx + line_len + 1
        line_range = lines_idx[current_idx + 1: next_idx]

        lines += [lines_vertices[line_range]]
        current_idx = next_idx
    return lines


def get_polydata_triangles(polydata):
    """ get triangles (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        triangles
    """
    vtk_polys = ns.vtk_to_numpy(polydata.GetPolys().GetData())
    assert((vtk_polys[::4] == 3).all())  # test if its really triangles
    return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T


def get_polydata_vertices(polydata):
    """ get vertices (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        points, represented as 2D ndarrays
    """
    return ns.vtk_to_numpy(polydata.GetPoints().GetData())


def get_polydata_normals(polydata):
    """ get vertices normal (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        Normals, represented as 2D ndarrays (Nx3). None if there are no normals
        in the vtk polydata.
    """
    vtk_normals = polydata.GetPointData().GetNormals()
    if vtk_normals is None:
        return None
    else:
        return ns.vtk_to_numpy(vtk_normals)


def get_polydata_colors(polydata):
    """ get points color (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    output : array (N, 3)
        Colors. None if no normals in the vtk polydata.
    """
    vtk_colors = polydata.GetPointData().GetScalars()
    if vtk_colors is None:
        return None
    else:
        return ns.vtk_to_numpy(vtk_colors)


def set_polydata_triangles(polydata, triangles):
    """ set polydata triangles with a numpy array (ndarrays Nx3 int)

    Parameters
    ----------
    polydata : vtkPolyData
    triangles : array (N, 3)
        triangles, represented as 2D ndarrays (Nx3)
    """
    vtk_triangles = np.hstack(np.c_[np.ones(len(triangles)).astype(np.int) * 3,
                                    triangles])
    vtk_triangles = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(len(triangles), vtk_triangles)
    polydata.SetPolys(vtk_cells)
    return polydata


def set_polydata_vertices(polydata, vertices):
    """ set polydata vertices with a numpy array (ndarrays Nx3 int)

    Parameters
    ----------
    polydata : vtkPolyData
    vertices : vertices, represented as 2D ndarrays (Nx3)
    """
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(vertices, deep=True))
    polydata.SetPoints(vtk_points)
    return polydata


def set_polydata_normals(polydata, normals):
    """ set polydata normals with a numpy array (ndarrays Nx3 int)

    Parameters
    ----------
    polydata : vtkPolyData
    normals : normals, represented as 2D ndarrays (Nx3) (one per vertex)
    """
    vtk_normals = ns.numpy_to_vtk(normals, deep=True)
    polydata.GetPointData().SetNormals(vtk_normals)
    return polydata


def set_polydata_colors(polydata, colors):
    """ set polydata colors with a numpy array (ndarrays Nx3 int)

    Parameters
    ----------
    polydata : vtkPolyData
    colors : colors, represented as 2D ndarrays (Nx3)
        colors are uint8 [0,255] RGB for each points
    """
    vtk_colors = ns.numpy_to_vtk(colors, deep=True,
                                 array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName("RGB")
    polydata.GetPointData().SetScalars(vtk_colors)
    return polydata


def update_polydata_normals(polydata):
    """ generate and update polydata normals

    Parameters
    ----------
    polydata : vtkPolyData
    """
    normals_gen = set_input(vtk.vtkPolyDataNormals(), polydata)
    normals_gen.ComputePointNormalsOn()
    normals_gen.ComputeCellNormalsOn()
    normals_gen.SplittingOff()
    # normals_gen.FlipNormalsOn()
    # normals_gen.ConsistencyOn()
    # normals_gen.AutoOrientNormalsOn()
    normals_gen.Update()

    vtk_normals = normals_gen.GetOutput().GetPointData().GetNormals()
    polydata.GetPointData().SetNormals(vtk_normals)


def get_polymapper_from_polydata(polydata):
    """ get vtkPolyDataMapper from a vtkPolyData

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    poly_mapper : vtkPolyDataMapper
    """
    poly_mapper = set_input(vtk.vtkPolyDataMapper(), polydata)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.InterpolateScalarsBeforeMappingOn()
    poly_mapper.Update()
    poly_mapper.StaticOn()
    return poly_mapper


def get_actor_from_polymapper(poly_mapper, light=(0.1, 0.15, 0.05)):
    """ get vtkActor from a vtkPolyDataMapper

    Parameters
    ----------
    poly_mapper : vtkPolyDataMapper

    Returns
    -------
    actor : vtkActor
    """
    actor = vtk.vtkActor()
    actor.SetMapper(poly_mapper)
    # actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetInterpolationToPhong()
    # actor.GetProperty().SetInterpolationToFlat()

    actor.GetProperty().SetAmbient(light[0])  # .3
    actor.GetProperty().SetDiffuse(light[1])  # .3
    actor.GetProperty().SetSpecular(light[2])  # .3
    return actor


def get_actor_from_polydata(polydata):
    """ get vtkActor from a vtkPolyData

    Parameters
    ----------
    polydata : vtkPolyData

    Returns
    -------
    actor : vtkActor
    """
    poly_mapper = get_polymapper_from_polydata(polydata)
    return get_actor_from_polymapper(poly_mapper)
