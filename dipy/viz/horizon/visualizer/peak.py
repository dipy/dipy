import warnings
from os.path import join as pjoin

import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury', min_version="0.9.0")
if has_fury:
    from fury.colormap import colormap_lookup_table
    from fury.lib import (
        VTK_OBJECT,
        Actor,
        CellArray,
        Command,
        PolyData,
        PolyDataMapper,
        calldata_type,
        numpy_support,
    )
    from fury.shaders import (
        attribute_to_actor,
        compose_shader,
        import_fury_shader,
        shader_to_actor,
    )
    from fury.utils import (
        apply_affine,
        numpy_to_vtk_colors,
        numpy_to_vtk_points
    )
else:
    class Actor:
        pass

    def calldata_type(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def VTK_OBJECT(*args):
        pass


class PeakActor(Actor):
    """FURY actor for visualizing DWI peaks.

    Parameters
    ----------
    directions : ndarray
        Peak directions. The shape of the array should be (X, Y, Z, D, 3).
    indices : tuple
        Indices given in tuple(x_indices, y_indices, z_indices)
        format for mapping 2D ODF array to 3D voxel grid.
    values : ndarray, optional
        Peak values. The shape of the array should be (X, Y, Z, D).
    affine : array, optional
        4x4 transformation array from native coordinates to world coordinates.
    colors : None or string ('rgb_standard') or tuple (3D or 4D) or
             array/ndarray (N, 3 or 4) or array/ndarray (K, 3 or 4) or
             array/ndarray(N, ) or array/ndarray (K, )
        If None a standard orientation colormap is used for every line.
        If one tuple of color is used. Then all streamlines will have the same
        color.
        If an array (N, 3 or 4) is given, where N is equal to the number of
        points. Then every point is colored with a different RGB(A) color.
        If an array (K, 3 or 4) is given, where K is equal to the number of
        lines. Then every line is colored with a different RGB(A) color.
        If an array (N, ) is given, where N is the number of points then these
        are considered as the values to be used by the colormap.
        If an array (K,) is given, where K is the number of lines then these
        are considered as the values to be used by the colormap.
    lookup_colormap : vtkLookupTable, optional
        Add a default lookup table to the colormap. Default is None which calls
        :func:`fury.actor.colormap_lookup_table`.
    linewidth : float, optional
        Line thickness. Default is 1.
    symmetric: bool, optional
        If True, peaks are drawn for both peaks_dirs and -peaks_dirs. Else,
        peaks are only drawn for directions given by peaks_dirs. Default is
        True.

    """

    def __init__(
        self,
        directions,
        indices,
        values=None,
        affine=None,
        colors=None,
        lookup_colormap=None,
        linewidth=1,
        symmetric=True,
    ):

        if affine is not None:
            w_pos = apply_affine(affine, np.asarray(indices).T)

        valid_dirs = directions[indices]

        num_dirs = len(np.where(valid_dirs >= 0)[0])

        pnts_per_line = 2

        points_array = np.empty((num_dirs * pnts_per_line, 3))
        centers_array = np.empty_like(points_array, dtype=int)
        diffs_array = np.empty_like(points_array)
        line_count = 0
        for idx, center in enumerate(zip(indices[0], indices[1], indices[2])):
            if affine is None:
                xyz = np.asarray(center)
            else:
                xyz = w_pos[idx, :]
            valid_peaks = np.where(valid_dirs[idx, :, :] >= 0)[0]
            for direction in valid_peaks:
                if values is not None:
                    pv = values[center][direction]
                else:
                    pv = 1.0

                if symmetric:
                    point_i = directions[center][direction] * pv + xyz
                    point_e = -directions[center][direction] * pv + xyz
                else:
                    point_i = directions[center][direction] * pv + xyz
                    point_e = xyz

                diff = point_e - point_i
                points_array[line_count * pnts_per_line, :] = point_e
                points_array[line_count * pnts_per_line + 1, :] = point_i
                centers_array[line_count * pnts_per_line, :] = center
                centers_array[line_count * pnts_per_line + 1, :] = center
                diffs_array[line_count * pnts_per_line, :] = diff
                diffs_array[line_count * pnts_per_line + 1, :] = diff
                line_count += 1

        vtk_points = numpy_to_vtk_points(points_array)

        vtk_cells = _points_to_vtk_cells(points_array)

        colors_tuple = _peaks_colors_from_points(points_array, colors=colors)
        vtk_colors, colors_are_scalars, self.__global_opacity = colors_tuple

        poly_data = PolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(vtk_cells)
        poly_data.GetPointData().SetScalars(vtk_colors)

        self.__mapper = PolyDataMapper()
        self.__mapper.SetInputData(poly_data)
        self.__mapper.ScalarVisibilityOn()
        self.__mapper.SetScalarModeToUsePointFieldData()
        self.__mapper.SelectColorArray('colors')
        self.__mapper.Update()

        self.SetMapper(self.__mapper)

        attribute_to_actor(self, centers_array, 'center')
        attribute_to_actor(self, diffs_array, 'diff')

        vs_var_dec = """
            in vec3 center;
            in vec3 diff;
            flat out vec3 centerVertexMCVSOutput;
            """
        fs_var_dec = """
            flat in vec3 centerVertexMCVSOutput;
            uniform bool isRange;
            uniform vec3 crossSection;
            uniform vec3 lowRanges;
            uniform vec3 highRanges;
            """
        orient_to_rgb = import_fury_shader(pjoin('utils',
                                                 'orient_to_rgb.glsl'))
        visible_cross_section = import_fury_shader(
            pjoin('interaction', 'visible_cross_section.glsl')
        )
        visible_range = import_fury_shader(pjoin('interaction',
                                                 'visible_range.glsl'))

        vs_dec = compose_shader([vs_var_dec, orient_to_rgb])
        fs_dec = compose_shader([fs_var_dec, visible_cross_section,
                                 visible_range])

        vs_impl = """
            centerVertexMCVSOutput = center;
            if (vertexColorVSOutput.rgb == vec3(0))
            {
                vertexColorVSOutput.rgb = orient2rgb(diff);
            }
            """

        fs_impl = """
            if (isRange)
            {
                if (!inVisibleRange(centerVertexMCVSOutput))
                    discard;
            }
            else
            {
                if (!inVisibleCrossSection(centerVertexMCVSOutput))
                    discard;
            }
            """

        shader_to_actor(self, 'vertex', decl_code=vs_dec, impl_code=vs_impl)
        shader_to_actor(self, 'fragment', decl_code=fs_dec)
        shader_to_actor(self, 'fragment', impl_code=fs_impl, block='light')

        # Color scale with a lookup table
        if colors_are_scalars:
            if lookup_colormap is None:
                lookup_colormap = colormap_lookup_table()

            self.__mapper.SetLookupTable(lookup_colormap)
            self.__mapper.UseLookupTableScalarRangeOn()
            self.__mapper.Update()

        self.__lw = linewidth
        self.GetProperty().SetLineWidth(self.__lw)

        if self.__global_opacity >= 0:
            self.GetProperty().SetOpacity(self.__global_opacity)

        self.__min_centers = np.min(indices, axis=1)
        self.__max_centers = np.max(indices, axis=1)

        self.__is_range = True
        self.__low_ranges = self.__min_centers
        self.__high_ranges = self.__max_centers
        self.__cross_section = self.__high_ranges // 2

        self.__mapper.AddObserver(
            Command.UpdateShaderEvent, self.__display_peaks_vtk_callback
        )

    @calldata_type(VTK_OBJECT)
    def __display_peaks_vtk_callback(self, caller, event, calldata=None):
        if calldata is not None:
            calldata.SetUniformi('isRange', self.__is_range)
            calldata.SetUniform3f('highRanges', self.__high_ranges)
            calldata.SetUniform3f('lowRanges', self.__low_ranges)
            calldata.SetUniform3f('crossSection', self.__cross_section)

    def display_cross_section(self, x, y, z):
        if self.__is_range:
            self.__is_range = False
        self.__cross_section = [x, y, z]

    def display_extent(self, x1, x2, y1, y2, z1, z2):
        if not self.__is_range:
            self.__is_range = True
        self.__low_ranges = [x1, y1, z1]
        self.__high_ranges = [x2, y2, z2]

    @property
    def cross_section(self):
        return self.__cross_section

    @property
    def global_opacity(self):
        return self.__global_opacity

    @global_opacity.setter
    def global_opacity(self, opacity):
        self.__global_opacity = opacity
        self.GetProperty().SetOpacity(self.__global_opacity)

    @property
    def high_ranges(self):
        return self.__high_ranges

    @property
    def is_range(self):
        return self.__is_range

    @property
    def low_ranges(self):
        return self.__low_ranges

    @property
    def linewidth(self):
        return self.__lw

    @linewidth.setter
    def linewidth(self, linewidth):
        self.__lw = linewidth
        self.GetProperty().SetLineWidth(self.__lw)

    @property
    def max_centers(self):
        return self.__max_centers

    @property
    def min_centers(self):
        return self.__min_centers


def peak(
    peaks_dirs,
    peaks_values=None,
    mask=None,
    affine=None,
    colors=None,
    linewidth=1,
    lookup_colormap=None,
    symmetric=True,
):
    """Visualize peak directions as given from ``peaks_from_model``.

    Parameters
    ----------
    peaks_dirs : ndarray
        Peak directions. The shape of the array should be (X, Y, Z, D, 3).
    peaks_values : ndarray, optional
        Peak values. The shape of the array should be (X, Y, Z, D).
    affine : array, optional
        4x4 transformation array from native coordinates to world coordinates.
    mask : ndarray, optional
        3D mask
    colors : tuple or None, optional
        Default None. If None then every peak gets an orientation color
        in similarity to a DEC map.
    lookup_colormap : vtkLookupTable, optional
        Add a default lookup table to the colormap. Default is None which calls
        :func:`fury.actor.colormap_lookup_table`.
    linewidth : float, optional
        Line thickness. Default is 1.
    symmetric : bool, optional
        If True, peaks are drawn for both peaks_dirs and -peaks_dirs. Else,
        peaks are only drawn for directions given by peaks_dirs. Default is
        True.

    Returns
    -------
    peak_actor : PeakActor
        Actor or LODActor representing the peaks directions and/or
        magnitudes.

    """
    if peaks_dirs.ndim != 5:
        raise ValueError(
            'Invalid peak directions. The shape of the structure '
            'must be (XxYxZxDx3). Your data has {} dimensions.'
            ''.format(peaks_dirs.ndim)
        )
    if peaks_dirs.shape[4] != 3:
        raise ValueError(
            'Invalid peak directions. The shape of the last '
            'dimension must be 3. Your data has a last dimension '
            'of {}.'.format(peaks_dirs.shape[4])
        )

    dirs_shape = peaks_dirs.shape

    if peaks_values is not None:
        if peaks_values.ndim != 4:
            raise ValueError(
                'Invalid peak values. The shape of the structure '
                'must be (XxYxZxD). Your data has {} dimensions.'
                ''.format(peaks_values.ndim)
            )
        vals_shape = peaks_values.shape
        if vals_shape != dirs_shape[:4]:
            raise ValueError(
                'Invalid peak values. The shape of the values '
                'must coincide with the shape of the directions.'
            )

    valid_mask = np.abs(peaks_dirs).max(axis=(-2, -1)) > 0
    if mask is not None:
        if mask.ndim != 3:
            warnings.warn(
                'Invalid mask. The mask must be a 3D array. The '
                'passed mask has {} dimensions. Ignoring passed '
                'mask.'.format(mask.ndim),
                UserWarning,
            )
        elif mask.shape != dirs_shape[:3]:
            warnings.warn(
                'Invalid mask. The shape of the mask must coincide '
                'with the shape of the directions. Ignoring passed '
                'mask.',
                UserWarning,
            )
        else:
            valid_mask = np.logical_and(valid_mask, mask)
    indices = np.where(valid_mask >= 0)

    return PeakActor(
        peaks_dirs,
        indices,
        values=peaks_values,
        affine=affine,
        colors=colors,
        lookup_colormap=lookup_colormap,
        linewidth=linewidth,
        symmetric=symmetric,
    )


def _peaks_colors_from_points(points, colors=None, points_per_line=2):
    """Return a VTK scalar array containing colors information for each one of
    the peaks according to the policy defined by the parameter colors.

    Parameters
    ----------
    points : (N, 3) array or ndarray
        points coordinates array.
    colors : None or string ('rgb_standard') or tuple (3D or 4D) or
             array/ndarray (N, 3 or 4) or array/ndarray (K, 3 or 4) or
             array/ndarray(N, ) or array/ndarray (K, )
        If None a standard orientation colormap is used for every line.
        If one tuple of color is used. Then all streamlines will have the same
        color.
        If an array (N, 3 or 4) is given, where N is equal to the number of
        points. Then every point is colored with a different RGB(A) color.
        If an array (K, 3 or 4) is given, where K is equal to the number of
        lines. Then every line is colored with a different RGB(A) color.
        If an array (N, ) is given, where N is the number of points then these
        are considered as the values to be used by the colormap.
        If an array (K,) is given, where K is the number of lines then these
        are considered as the values to be used by the colormap.
    points_per_line : int (1 or 2), optional
        number of points per peak direction.

    Returns
    -------
    color_array : vtkDataArray
        vtk scalar array with name 'colors'.
    colors_are_scalars : bool
        indicates whether or not the colors are scalars to be interpreted by a
        colormap.
    global_opacity : float
        returns 1 if the colors array doesn't contain opacity otherwise -1.

    """
    num_pnts = len(points)
    num_lines = num_pnts // points_per_line
    colors_are_scalars = False
    global_opacity = 1
    if colors is None or colors == 'rgb_standard':
        # Automatic RGB colors
        colors = np.asarray((0, 0, 0))
        color_array = numpy_to_vtk_colors(np.tile(255 * colors, (num_pnts, 1)))
    elif type(colors) is tuple:
        global_opacity = 1 if len(colors) == 3 else -1
        colors = np.asarray(colors)
        color_array = numpy_to_vtk_colors(np.tile(255 * colors, (num_pnts, 1)))
    else:
        colors = np.asarray(colors)
        if len(colors) == num_lines:
            pnts_colors = np.repeat(colors, points_per_line, axis=0)
            if colors.ndim == 1:  # Scalar per line
                color_array = numpy_support.numpy_to_vtk(pnts_colors,
                                                         deep=True)
                colors_are_scalars = True
            elif colors.ndim == 2:  # RGB(A) color per line
                global_opacity = 1 if colors.shape[1] == 3 else -1
                color_array = numpy_to_vtk_colors(255 * pnts_colors)
        elif len(colors) == num_pnts:
            if colors.ndim == 1:  # Scalar per point
                color_array = numpy_support.numpy_to_vtk(colors, deep=True)
                colors_are_scalars = True
            elif colors.ndim == 2:  # RGB(A) color per point
                global_opacity = 1 if colors.shape[1] == 3 else -1
                color_array = numpy_to_vtk_colors(255 * colors)

    color_array.SetName('colors')
    return color_array, colors_are_scalars, global_opacity


def _points_to_vtk_cells(points, points_per_line=2):
    """Return the VTK cell array for the peaks given the set of points
    coordinates.

    Parameters
    ----------
    points : (N, 3) array or ndarray
        points coordinates array.
    points_per_line : int (1 or 2), optional
        number of points per peak direction.

    Returns
    -------
    cell_array : vtkCellArray
        connectivity + offset information.

    """
    num_pnts = len(points)
    num_cells = num_pnts // points_per_line

    cell_array = CellArray()

    """
    Connectivity is an array that contains the indices of the points that
    need to be connected in the visualization. The indices start from 0.
    """
    connectivity = np.asarray(list(range(0, num_pnts)), dtype=int)
    """
    Offset is an array that contains the indices of the first point of
    each line. The indices start from 0 and given the known geometry of
    this actor the creation of this array requires a 2 points padding
    between indices.
    """
    offset = np.asarray(list(range(0, num_pnts + 1, points_per_line)),
                        dtype=int)

    vtk_array_type = numpy_support.get_vtk_array_type(connectivity.dtype)
    cell_array.SetData(
        numpy_support.numpy_to_vtk(offset, deep=True,
                                   array_type=vtk_array_type),
        numpy_support.numpy_to_vtk(connectivity, deep=True,
                                   array_type=vtk_array_type),
    )

    cell_array.SetNumberOfCells(num_cells)
    return cell_array


class PeaksVisualizer:
    def __init__(self, pam, world_coords):
        self._peak_dirs, self._affine = pam
        if world_coords:
            self._peak_actor = peak(self._peak_dirs, affine=self._affine)
        else:
            self._peak_actor = peak(self._peak_dirs)

    @property
    def actors(self):
        return [self._peak_actor]
