# from __future__ import division, print_function, absolute_import

import numpy as np
from nibabel.affines import apply_affine

from dipy.viz.colormap import line_colors
from dipy.viz.utils import numpy_to_vtk_points, numpy_to_vtk_colors
from dipy.viz.utils import vtk_matrix_to_numpy, numpy_to_vtk_matrix
from dipy.viz.utils import set_input, map_coordinates_3d_4d, get_bounding_box_sizes
from dipy.viz import layout
from dipy.core.ndindex import ndindex

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package
from dipy.utils.six import string_types

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

if have_vtk:

    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()


def slicer(data, affine=None, value_range=None, opacity=1.,
           lookup_colormap=None):
    """ Cuts 3D scalar or rgb volumes into images

    Parameters
    ----------
    data : array, shape (X, Y, Z) or (X, Y, Z, 3)
        A grayscale or rgb 4D volume as a numpy array.
    affine : array, shape (3, 3)
        Grid to space (usually RAS 1mm) transformation matrix
    value_range : None or tuple (2,)
        If None then the values will be interpolated to (0, 255) from
        (min, max). Otherwise from (value_range[0], value_range[1]).
    opacity : float
        Opacity of 0 means completely transparent and 1 completely visible.
    lookup_colormap : vtkLookupTable
        If None (default) then a grayscale map is created.

    Returns
    -------
    image_actor : ImageActor
        An object that is capable of displaying different parts of the volume
        as slices. The key method of this object is ``display_extent`` where
        one can input grid coordinates and display the slice in space (or grid)
        coordinates as calculated by the affine parameter.

    """
    if data.ndim != 3:
        if data.ndim == 4:
            if data.shape[3] != 3:
                raise ValueError('Only RGB 3D arrays are currently supported.')
            else:
                nb_components = 3
        else:
            raise ValueError('Only 3D arrays are currently supported.')
    else:
        nb_components = 1

    if value_range is None:
        vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
    else:
        vol = np.interp(data, xp=[value_range[0], value_range[1]], fp=[0, 255])
    vol = vol.astype('uint8')

    im = vtk.vtkImageData()
    if major_version <= 5:
        im.SetScalarTypeToUnsignedChar()
    I, J, K = vol.shape[:3]
    im.SetDimensions(I, J, K)
    voxsz = (1., 1., 1.)
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    if major_version <= 5:
        im.AllocateScalars()
        im.SetNumberOfScalarComponents(nb_components)
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, nb_components)

    # copy data
    # what I do below is the same as what is commented here but much faster
    # for index in ndindex(vol.shape):
    #     i, j, k = index
    #     im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)

    if nb_components == 1:
        vol = vol.ravel()
    else:
        vol = np.reshape(vol, [np.prod(vol.shape[:3]), vol.shape[3]])

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    if affine is None:
        affine = np.eye(4)

    # Set the transform (identity if none given)
    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()
    transform_matrix.DeepCopy((
        affine[0][0], affine[0][1], affine[0][2], affine[0][3],
        affine[1][0], affine[1][1], affine[1][2], affine[1][3],
        affine[2][0], affine[2][1], affine[2][2], affine[2][3],
        affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    # Set the reslicing
    image_resliced = vtk.vtkImageReslice()
    set_input(image_resliced, im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()
    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    if nb_components == 1:
        if lookup_colormap is None:
            # Create a black/white lookup table.
            lut = colormap_lookup_table((0, 255), (0, 0), (0, 0), (0, 1))
        else:
            lut = lookup_colormap

    x1, x2, y1, y2, z1, z2 = im.GetExtent()
    ex1, ex2, ey1, ey2, ez1, ez2 = image_resliced.GetOutput().GetExtent()

    class ImageActor(vtk.vtkImageActor):

        def input_connection(self, output):
            if vtk.VTK_MAJOR_VERSION <= 5:
                self.SetInput(output.GetOutput())
            else:
                self.GetMapper().SetInputConnection(output.GetOutputPort())
            self.output = output
            self.shape = (ex2 + 1, ey2 + 1, ez2 + 1)

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            self.SetDisplayExtent(x1, x2, y1, y2, z1, z2)
            if vtk.VTK_MAJOR_VERSION > 5:
                self.Update()

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(ex1, ex2, ey1, ey2, ez2/2, ez2/2)
            if x is not None:
                self.display_extent(x, x, ey1, ey2, ez1, ez2)
            if y is not None:
                self.display_extent(ex1, ex2, y, y, ez1, ez2)
            if z is not None:
                self.display_extent(ex1, ex2, ey1, ey2, z, z)

        def opacity(self, value):
            if vtk.VTK_MAJOR_VERSION <= 5:
                self.SetOpacity(value)
            else:
                self.GetProperty().SetOpacity(value)

        def copy(self):
            im_actor = ImageActor()
            im_actor.input_connection(self.output)
            im_actor.SetDisplayExtent(*self.GetDisplayExtent())
            im_actor.opacity(opacity)
            return im_actor

        def set_position(self, wx, wy, wz):
            self.SetPosition(wx, wy, wz)

        def get_position(self):
            return self.GetPosition()

        def get_data(self):
            data = self.output.GetOutput()
            scalars = data.GetPointData().GetScalars()
            data = numpy_support.vtk_to_numpy(scalars)
            shape = self.shape
            if data.ndim == 2:
                data = data.reshape(shape[2], shape[1], shape[0],
                                    data.shape[-1])
            else:
                data = data.reshape(shape[2], shape[1], shape[0])
            return data.swapaxes(0, 2)

    image_actor = ImageActor()
    if nb_components == 1:
        plane_colors = vtk.vtkImageMapToColors()
        plane_colors.SetLookupTable(lut)
        plane_colors.SetInputConnection(image_resliced.GetOutputPort())
        plane_colors.Update()
        image_actor.input_connection(plane_colors)
    else:
        image_actor.input_connection(image_resliced)
    image_actor.display()
    image_actor.opacity(opacity)
    image_actor.GetMapper().BorderOn()

    return image_actor


def odf_slicer(odfs, affine=None, mask=None, sphere=None, scale=2.2,
               norm=True, radial_scale=True, opacity=1.,
               colormap=None):
    """ Slice spherical fields
    """

    if mask is None:
        mask = np.ones(odfs.shape[:3], dtype=np.bool)
    else:
        mask = mask.astype(np.bool)

    class OdfSlicerActor(vtk.vtkLODActor):

        def display_extent(self, x1, x2, y1, y2, z1, z2):

            tmp_mask = np.zeros(odfs.shape[:3], dtype=np.bool)
            tmp_mask[x1:x2, y1:y2, z1:z2] = True

            tmp_mask = np.bitwise_and(tmp_mask, mask)

            self.mapper = _odf_slicer_mapper(odfs=odfs,
                                             affine=affine,
                                             mask=tmp_mask,
                                             sphere=sphere,
                                             scale=scale,
                                             norm=norm,
                                             radial_scale=radial_scale,
                                             opacity=opacity,
                                             colormap=colormap)
            self.SetMapper(self.mapper)

    odf_actor = OdfSlicerActor()
    I, J, K = odfs.shape[:3]
    odf_actor.display_extent(0, I, 0, J, K/2, K/2 + 1)

    return odf_actor


def _odf_slicer_mapper(odfs, affine=None, mask=None, sphere=None, scale=2.2,
                       norm=True, radial_scale=True, opacity=1.,
                       colormap=None):

    if mask is None:
        mask = np.ones(odfs.shape[:3])

    ijk = np.ascontiguousarray(np.array(np.nonzero(mask)).T)

    if affine is not None:
        ijk = np.ascontiguousarray(apply_affine(affine, ijk))

    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    all_xyz = []
    all_faces = []
    all_ms = []
    for (k, center) in enumerate(ijk):
        m = odfs[tuple(center)].copy()

        if norm:
            m /= abs(m).max()

        if radial_scale:
            xyz = vertices * m[:, None]
        else:
            xyz = vertices.copy()

        all_xyz.append(scale * xyz + center)
        all_faces.append(faces + k * xyz.shape[0])
        all_ms.append(m)

    all_xyz = np.ascontiguousarray(np.concatenate(all_xyz))
    all_xyz_vtk = numpy_support.numpy_to_vtk(all_xyz, deep=True)

    all_faces = np.concatenate(all_faces)
    all_faces = np.hstack((3 * np.ones((len(all_faces), 1)),
                           all_faces))
    ncells = len(all_faces)

    all_faces = np.ascontiguousarray(all_faces.ravel(), dtype='i8')
    all_faces_vtk = numpy_support.numpy_to_vtkIdTypeArray(all_faces,
                                                          deep=True)

    all_ms = np.ascontiguousarray(np.concatenate(all_ms), dtype='f4')

    points = vtk.vtkPoints()
    points.SetData(all_xyz_vtk)

    cells = vtk.vtkCellArray()
    cells.SetCells(ncells, all_faces_vtk)

    if colormap is not None:
        from dipy.viz.fvtk import create_colormap
        cols = create_colormap(all_ms.ravel(), colormap)
        # cols = np.interp(cols, [0, 1], [0, 255]).astype('ubyte')

        # vtk_colors = numpy_to_vtk_colors(255 * cols)

        vtk_colors = numpy_support.numpy_to_vtk(
            np.asarray(255 * cols),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR)

        vtk_colors.SetName("Colors")

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    if colormap is not None:
        polydata.GetPointData().SetScalars(vtk_colors)

    mapper = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)

    return mapper


def streamtube(lines, colors=None, opacity=1, linewidth=0.01, tube_sides=9,
               lod=True, lod_points=10 ** 4, lod_points_size=3,
               spline_subdiv=None, lookup_colormap=None):
    """ Uses streamtubes to visualize polylines

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
        If an array (K, ) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L, ) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    opacity : float
    linewidth : float
    tube_sides : int
    lod : bool
        use vtkLODActor rather than vtkActor
    lod_points : int
        number of points to be used when LOD is in effect
    lod_points_size : int
        size of points when lod is in effect
    spline_subdiv : int
        number of splines subdivision to smooth streamtubes
    lookup_colormap : bool
        add a default lookup table to the colormap

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10, 3), np.random.rand(20, 3)]
    >>> colors=np.random.rand(2, 3)
    >>> c=fvtk.streamtube(lines, colors)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)

    Notes
    -----
    Streamtubes can be heavy on GPU when loading many streamlines and
    therefore, you may experience slow rendering time depending on system GPU.
    A solution to this problem is to reduce the number of points in each
    streamline. In Dipy we provide an algorithm that will reduce the number of
    points on the straighter parts of the streamline but keep more points on
    the curvier parts. This can be used in the following way

    from dipy.tracking.distances import approx_polygon_track
    lines = [approx_polygon_track(line, 0.2) for line in lines]

    Alternatively we suggest using the ``line`` actor which is much more
    efficient.

    See Also
    --------
    dipy.viz.actor.line
    """
    # Poly data with lines and colors
    poly_data, is_colormap = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # Set Normals
    poly_normals = set_input(vtk.vtkPolyDataNormals(), next_input)
    poly_normals.ComputeCellNormalsOn()
    poly_normals.ComputePointNormalsOn()
    poly_normals.ConsistencyOn()
    poly_normals.AutoOrientNormalsOn()
    poly_normals.Update()
    next_input = poly_normals.GetOutputPort()

    # Spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0):
        spline_filter = set_input(vtk.vtkSplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    # Add thickness to the resulting lines
    tube_filter = set_input(vtk.vtkTubeFilter(), next_input)
    tube_filter.SetNumberOfSides(tube_sides)
    tube_filter.SetRadius(linewidth)
    # tube_filter.SetVaryRadiusToVaryRadiusByScalar()
    tube_filter.CappingOn()
    tube_filter.Update()
    next_input = tube_filter.GetOutputPort()

    # Poly mapper
    poly_mapper = set_input(vtk.vtkPolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("Colors")
    poly_mapper.GlobalImmediateModeRenderingOn()
    # poly_mapper.SetColorModeToMapScalars()
    poly_mapper.Update()

    # Color Scale with a lookup table
    if is_colormap:
        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()
        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    if lod:
        actor = vtk.vtkLODActor()
        actor.SetNumberOfCloudPoints(lod_points)
        actor.GetProperty().SetPointSize(lod_points_size)
    else:
        actor = vtk.vtkActor()

    actor.SetMapper(poly_mapper)
    actor.GetProperty().SetAmbient(0.1)  # .3
    actor.GetProperty().SetDiffuse(0.15)  # .3
    actor.GetProperty().SetSpecular(0.05)  # .3
    actor.GetProperty().SetSpecularPower(6)
    # actor.GetProperty().SetInterpolationToGouraud()
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetOpacity(opacity)

    return actor


def line(lines, colors=None, opacity=1, linewidth=1,
         spline_subdiv=None, lod=True, lod_points=10 ** 4, lod_points_size=3,
         lookup_colormap=None):
    """ Create an actor for one or more lines.

    Parameters
    ------------
    lines :  list of arrays

    colors : array (N, 3), list of arrays, tuple (3,), array (K,), None
        If None then a standard orientation colormap is used for every line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K, ) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L, ) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    opacity : float, optional

    linewidth : float, optional
        Line thickness.
    spline_subdiv : int, optional
        number of splines subdivision to smooth streamtubes
    lookup_colormap : bool, optional
        add a default lookup table to the colormap

    Returns
    ----------
    v : vtkActor object
        Line.

    Examples
    ----------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10,3), np.random.rand(20,3)]
    >>> colors=np.random.rand(2,3)
    >>> c=fvtk.line(lines, colors)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)
    """
    # Poly data with lines and colors
    poly_data, is_colormap = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # use spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0):
        spline_filter = set_input(vtk.vtkSplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    poly_mapper = set_input(vtk.vtkPolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("Colors")
    # poly_mapper.SetColorModeToMapScalars()
    poly_mapper.Update()

    # Color Scale with a lookup table
    if is_colormap:

        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()

        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    if lod:
        actor = vtk.vtkLODActor()
        actor.SetNumberOfCloudPoints(lod_points)
        actor.GetProperty().SetPointSize(lod_points_size)
    else:
        actor = vtk.vtkActor()

    # actor = vtk.vtkActor()
    actor.SetMapper(poly_mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetOpacity(opacity)

    return actor


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
        If an array (K, ) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L, ) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.

    Returns
    -------
    poly_data :  vtkPolyData
    is_colormap : bool, true if the input color array was a colormap
    """

    # Get the 3d points_array
    points_array = np.vstack(lines)

    nb_lines = len(lines)
    nb_points = len(points_array)

    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    lines_array = []
    points_per_line = np.zeros([nb_lines], np.int64)
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
    vtk_lines.GetData().DeepCopy(numpy_support.numpy_to_vtk(lines_array))
    vtk_lines.SetNumberOfCells(len(lines))

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
                vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

            elif cols_arr.ndim == 1:
                if len(cols_arr) == nb_lines:  # values for every streamline
                    cols_arrx = []
                    for (i, value) in enumerate(colors):
                        cols_arrx += lines[i].shape[0]*[value]
                    cols_arrx = np.array(cols_arrx)
                    vtk_colors = numpy_support.numpy_to_vtk(cols_arrx,
                                                            deep=True)
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
                vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

    vtk_colors.SetName("Colors")

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)
    poly_data.GetPointData().SetScalars(vtk_colors)
    return poly_data, is_colormap


def colormap_lookup_table(scale_range=(0, 1), hue_range=(0.8, 0),
                          saturation_range=(1, 1), value_range=(0.8, 0.8)):
    """ Lookup table for the colormap

    Parameters
    ----------
    scale_range : tuple
        It can be anything e.g. (0, 1) or (0, 255). Usually it is the mininum
        and maximum value of your data.
    hue_range : tuple of floats
        HSV values (min 0 and max 1)
    saturation_range : tuple of floats
        HSV values (min 0 and max 1)
    value_range : tuple of floats
        HSV value (min 0 and max 1)

    Returns
    -------
    lookup_table : vtkLookupTable

    """
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetRange(scale_range)
    lookup_table.SetTableRange(scale_range)

    lookup_table.SetHueRange(hue_range)
    lookup_table.SetSaturationRange(saturation_range)
    lookup_table.SetValueRange(value_range)

    lookup_table.Build()
    return lookup_table


def scalar_bar(lookup_table=None, title=" "):
    """ Default scalar bar actor for a given colormap

    Parameters
    ----------
    lookup_table : vtkLookupTable or None
        If None then ``colormap_lookup_table`` is called with default options.
    title : str

    Returns
    -------
    scalar_bar : vtkScalarBarActor

    """
    lookup_table_copy = vtk.vtkLookupTable()
    # Deepcopy the lookup_table because sometimes vtkPolyDataMapper deletes it
    if lookup_table is None:
        lookup_table = colormap_lookup_table()
    lookup_table_copy.DeepCopy(lookup_table)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetTitle(title)
    scalar_bar.SetLookupTable(lookup_table_copy)
    scalar_bar.SetNumberOfLabels(6)

    return scalar_bar


def _arrow(pos=(0, 0, 0), color=(1, 0, 0), scale=(1, 1, 1), opacity=1):
    ''' Internal function for generating arrow actors.
    '''
    arrow = vtk.vtkArrowSource()
    # arrow.SetTipLength(length)

    arrowm = vtk.vtkPolyDataMapper()

    if major_version <= 5:
        arrowm.SetInput(arrow.GetOutput())
    else:
        arrowm.SetInputConnection(arrow.GetOutputPort())

    arrowa = vtk.vtkActor()
    arrowa.SetMapper(arrowm)

    arrowa.GetProperty().SetColor(color)
    arrowa.GetProperty().SetOpacity(opacity)
    arrowa.SetScale(scale)

    return arrowa


def axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1),
         opacity=1):
    """ Create an actor with the coordinate's system axes where
    red = x, green = y, blue = z.

    Parameters
    ----------
    scale : tuple (3,)
        axes size e.g. (100, 100, 100)
    colorx : tuple (3,)
        x-axis color. Default red.
    colory : tuple (3,)
        y-axis color. Default blue.
    colorz : tuple (3,)
        z-axis color. Default green.

    Returns
    -------
    vtkAssembly

    """

    arrowx = _arrow(color=colorx, scale=scale, opacity=opacity)
    arrowy = _arrow(color=colory, scale=scale, opacity=opacity)
    arrowz = _arrow(color=colorz, scale=scale, opacity=opacity)

    arrowy.RotateZ(90)
    arrowz.RotateY(-90)

    ass = vtk.vtkAssembly()
    ass.AddPart(arrowx)
    ass.AddPart(arrowy)
    ass.AddPart(arrowz)

    return ass


def text_overlay(text, position=(0, 0), color=(1, 1, 1),
                 font_size=12, font_family='Arial', justification='left',
                 bold=False, italic=False, shadow=False):

    class TextActor(vtk.vtkTextActor):

        def message(self, text):
            self.SetInput(text)

        def set_message(self, text):
            self.SetInput(text)

        def get_message(self):
            return self.GetInput()

        def font_size(self, size):
            self.GetTextProperty().SetFontSize(size)

        def font_family(self, family='Arial'):
            self.GetTextProperty().SetFontFamilyToArial()

        def justification(self, justification):
            tprop = self.GetTextProperty()
            if justification == 'left':
                tprop.SetJustificationToLeft()
            if justification == 'center':
                tprop.SetJustificationToCentered()
            if justification == 'right':
                tprop.SetJustificationToRight()

        def font_style(self, bold=False, italic=False, shadow=False):
            tprop = self.GetTextProperty()
            if bold:
                tprop.BoldOn()
            else:
                tprop.BoldOff()
            if italic:
                tprop.ItalicOn()
            else:
                tprop.ItalicOff()
            if shadow:
                tprop.ShadowOn()
            else:
                tprop.ShadowOff()

        def color(self, color):
            self.GetTextProperty().SetColor(*color)

        def set_position(self, position):
            self.SetDisplayPosition(*position)

        def get_position(self, position):
            return self.GetDisplayPosition()

    text_actor = TextActor()
    text_actor.set_position(position)
    text_actor.message(text)
    text_actor.font_size(font_size)
    text_actor.font_family(font_family)
    text_actor.justification(justification)
    text_actor.font_style(bold, italic, shadow)
    text_actor.color(color)

    return text_actor


def text_3d(text, position=(0, 0, 0), color=(1, 1, 1),
            font_size=12, font_family='Arial', justification='left',
            vertical_justification="bottom",
            bold=False, italic=False, shadow=False):

    class TextActor3D(vtk.vtkTextActor3D):
        def message(self, text):
            self.set_message(text)

        def set_message(self, text):
            self.SetInput(text)
            self._update_user_matrix()

        def get_message(self):
            return self.GetInput()

        def font_size(self, size):
            self.GetTextProperty().SetFontSize(24)
            text_actor.SetScale((1./24.*size,)*3)
            self._update_user_matrix()

        def font_family(self, family='Arial'):
            self.GetTextProperty().SetFontFamilyToArial()
            #self._update_user_matrix()

        def justification(self, justification):
            tprop = self.GetTextProperty()
            if justification == 'left':
                tprop.SetJustificationToLeft()
            elif justification == 'center':
                tprop.SetJustificationToCentered()
            elif justification == 'right':
                tprop.SetJustificationToRight()
            else:
                raise ValueError("Unknown justification: '{}'".format(justification))

            self._update_user_matrix()

        def vertical_justification(self, justification):
            tprop = self.GetTextProperty()
            if justification == 'top':
                tprop.SetVerticalJustificationToTop()
            elif justification == 'middle':
                tprop.SetVerticalJustificationToCentered()
            elif justification == 'bottom':
                tprop.SetVerticalJustificationToBottom()
            else:
                raise ValueError("Unknown vertical justification: '{}'".format(justification))

            self._update_user_matrix()

        def font_style(self, bold=False, italic=False, shadow=False):
            tprop = self.GetTextProperty()
            if bold:
                tprop.BoldOn()
            else:
                tprop.BoldOff()
            if italic:
                tprop.ItalicOn()
            else:
                tprop.ItalicOff()
            if shadow:
                tprop.ShadowOn()
            else:
                tprop.ShadowOff()

            self._update_user_matrix()

        def color(self, color):
            self.GetTextProperty().SetColor(*color)

        def set_position(self, position):
            self.SetPosition(position)

        def get_position(self, position):
            return self.GetPosition()

        # def GetCenter(self):
        #     """ Replace parent function as it always returns (0,0,0). """
        #     return self.GetOrigin()

        # def GetOrigin(self):
        #     bounds = self.GetBounds()
        #     #center = self.GetCenter()
        #     X1, X2, Y1, Y2, Z1, Z2 = self.GetBounds()
        #     width, height, depth = X2-X1, Y2-Y1, Z2-Z1
        #     center = X1+width/2., Y1+height/2., Z1+depth/2.

        #     origin = np.zeros(3)
        #     tprop = self.GetTextProperty()
        #     if tprop.GetJustification() == vtk.VTK_TEXT_LEFT:
        #         origin[0] = bounds[0]
        #     elif tprop.GetJustification() == vtk.VTK_TEXT_CENTERED:
        #         origin[0] = center[0]
        #     elif tprop.GetJustification() == vtk.VTK_TEXT_RIGHT:
        #         origin[0] = bounds[1]

        #     if tprop.GetVerticalJustification() == vtk.VTK_TEXT_BOTTOM:
        #         origin[1] = bounds[2]
        #     elif tprop.GetVerticalJustification() == vtk.VTK_TEXT_CENTERED:
        #         origin[1] = center[1]
        #     elif tprop.GetVerticalJustification() == vtk.VTK_TEXT_TOP:
        #         origin[1] = bounds[3]

        #     return origin

        def _update_user_matrix(self):
            """
            Text justification of vtkTextActor3D doesn't seem to be working, so we do it manually.
            """
            user_matrix = np.eye(4)

            text_bounds = [0, 0, 0, 0]
            self.GetBoundingBox(text_bounds)

            tprop = self.GetTextProperty()
            if tprop.GetJustification() == vtk.VTK_TEXT_LEFT:
                user_matrix[:3, -1] += (-text_bounds[0], 0, 0)
            elif tprop.GetJustification() == vtk.VTK_TEXT_CENTERED:
                user_matrix[:3, -1] += (-(text_bounds[0]+(text_bounds[1]-text_bounds[0])/2.), 0, 0)
            elif tprop.GetJustification() == vtk.VTK_TEXT_RIGHT:
                user_matrix[:3, -1] += (-text_bounds[1], 0, 0)

            if tprop.GetVerticalJustification() == vtk.VTK_TEXT_BOTTOM:
                user_matrix[:3, -1] += (0, -text_bounds[2], 0)
            elif tprop.GetVerticalJustification() == vtk.VTK_TEXT_CENTERED:
                user_matrix[:3, -1] += (0, -(text_bounds[2]+(text_bounds[3]-text_bounds[2])/2), 0)
            elif tprop.GetVerticalJustification() == vtk.VTK_TEXT_TOP:
                user_matrix[:3, -1] += (0, -text_bounds[3], 0)

            user_matrix[:3, -1] *= self.GetScale()
            self.SetUserMatrix(numpy_to_vtk_matrix(user_matrix))

    text_actor = TextActor3D()
    text_actor.message(text)
    text_actor.font_size(font_size)
    text_actor.set_position(position)
    text_actor.font_family(font_family)
    text_actor.font_style(bold, italic, shadow)
    text_actor.color(color)
    text_actor.justification(justification)
    text_actor.vertical_justification(vertical_justification)

    return text_actor


def figure(pic, interpolation='nearest'):
    """ Return a figure as an image actor

    Parameters
    ----------
    pic : filename or numpy RGBA array

    interpolation : str
        Options are nearest, linear or cubic. Default is nearest.

    Returns
    -------
    image_actor : vtkImageActor
    """

    if isinstance(pic, string_types):
        png = vtk.vtkPNGReader()
        png.SetFileName(pic)
        png.Update()
        vtk_image_data = png.GetOutput()
    else:

        if pic.ndim == 3 and pic.shape[2] == 4:

            vtk_image_data = vtk.vtkImageData()
            if major_version <= 5:
                vtk_image_data.SetScalarTypeToUnsignedChar()

            if major_version <= 5:
                vtk_image_data.AllocateScalars()
                vtk_image_data.SetNumberOfScalarComponents(4)
            else:
                vtk_image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)

            # width, height
            vtk_image_data.SetDimensions(pic.shape[1], pic.shape[0], 1)
            vtk_image_data.SetExtent(0, pic.shape[1] - 1,
                                     0, pic.shape[0] - 1,
                                     0, 0)
            pic_tmp = np.swapaxes(pic, 0, 1)
            pic_tmp = pic.reshape(pic.shape[1] * pic.shape[0], 4)
            pic_tmp = np.ascontiguousarray(pic_tmp)
            uchar_array = numpy_support.numpy_to_vtk(pic_tmp, deep=True)
            vtk_image_data.GetPointData().SetScalars(uchar_array)

    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(vtk_image_data)

    if interpolation == 'nearest':
        image_actor.GetProperty().SetInterpolationTypeToNearest()

    if interpolation == 'linear':
        image_actor.GetProperty().SetInterpolationTypeToLinear()

    if interpolation == 'cubic':
        image_actor.GetProperty().SetInterpolationTypeToCubic()

    image_actor.Update()
    return image_actor


class Container(vtk.vtkAssembly):
    def __init__(self, layout=layout.Layout(), name=""):
        self.actors = []
        self.user_matrices = {}  # Backup
        self.layout = layout
        self.name = name
        self.anchor = np.zeros(3)

    def add(self, *actors):
        self.actors.extend(actors)

        for a in actors:
            self.AddPart(a)

            self.user_matrices[a] = vtk_matrix_to_numpy(a.GetUserMatrix())
            if isinstance(a, Container):
                self.user_matrices.update(a.user_matrices)

        self.layout.apply(actors)

    def get_actors(self):
        actors = []
        nb_paths = self.GetNumberOfPaths()
        self.InitPathTraversal()

        for i in range(nb_paths):
            path = self.GetNextPath()
            parent_container = path.GetItemAsObject(path.GetNumberOfItems()-2)
            obj = path.GetLastNode().GetViewProp()

            new_user_matrix = vtk_matrix_to_numpy(parent_container.GetMatrix())
            if self.user_matrices[obj] is not None:
                new_user_matrix = np.dot(new_user_matrix, self.user_matrices[obj])

            obj.SetUserMatrix(numpy_to_vtk_matrix(new_user_matrix))
            actors.append(obj)

        return actors

    def __len__(self):
        return len(self.actors)
