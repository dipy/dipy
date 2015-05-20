
#from __future__ import division, print_function, absolute_import

import numpy as np

from dipy.viz.colormap import line_colors
from dipy.viz.utils import numpy_to_vtk_points, numpy_to_vtk_colors
from dipy.viz.utils import set_input, trilinear_interp
from dipy.core.ndindex import ndindex

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

if have_vtk:

    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()


def slice(data, affine):
    """ Cuts 3D images

    Parameters
    ----------
    data : array, shape (X, Y, Z)
        A volume as a numpy array.
    affine : array, shape (3, 3)
        Grid to space (usually RAS 1mm) transformation matrix

    Returns
    -------
    vtkImageActor

    """

    vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
    vol = vol.astype('uint8')

    im = vtk.vtkImageData()
    if major_version <= 5:
        im.SetScalarTypeToUnsignedChar()
    I, J, K = vol.shape[:3]
    im.SetDimensions(I, J, K)
    voxsz = (1., 1., 1)
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    if major_version <= 5:
        im.AllocateScalars()
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    # copy data
    for index in ndindex(vol.shape):
        i, j, k = index
        im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

    # Set the transform (identity if none given)
    transform = vtk.vtkTransform()
    if affine is not None:
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
    image_resliced.SetInputData(im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()
    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    # Get back resliced image
    # im = image_data #image_resliced.GetOutput()

    # An outline provides context around the data.
    #    outline_data = vtk.vtkOutlineFilter()
    #    set_input(outline_data, im)
    #
    #    mapOutline = vtk.vtkPolyDataMapper()
    #    mapOutline.SetInputConnection(outline_data.GetOutputPort())
    #    outline_ = vtk.vtkActor()
    #    outline_.SetMapper(mapOutline)
    #    outline_.GetProperty().SetColor(1, 0, 0)

    # Now we are creating three orthogonal planes passing through the
    # volume. Each plane uses a different texture map and therefore has
    # diferent coloration.

    # Start by creatin a black/white lookup table.
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(0, 255)
    # print(data.min(), data.max())
    lut.SetSaturationRange(0, 0)
    lut.SetHueRange(0, 0)
    lut.SetValueRange(0, 1)
    lut.SetRampToLinear()
    lut.Build()

    x1, x2, y1, y2, z1, z2 = im.GetExtent()

    # Create the first of the three planes. The filter vtkImageMapToColors
    # maps the data through the corresponding lookup table created above.
    # The vtkImageActor is a type of vtkProp and conveniently displays an
    # image on a single quadrilateral plane. It does this using texture
    # mapping and as a result is quite fast. (Note: the input image has to
    # be unsigned char values, which the vtkImageMapToColors produces.)
    # Note also that by specifying the DisplayExtent, the pipeline
    # requests data of this extent and the vtkImageMapToColors only
    # processes a slice of data.
    plane_colors = vtk.vtkImageMapToColors()
    plane_colors.SetLookupTable(lut)
    plane_colors.SetInputConnection(image_resliced.GetOutputPort())
    plane_colors.Update()

    saggital = vtk.vtkImageActor()
    # set_input(saggital, plane_colors.GetOutput())
    saggital.GetMapper().SetInputConnection(plane_colors.GetOutputPort())
    #saggital.SetDisplayExtent(0, 0, y1, y2, z1, z2)
    saggital.SetDisplayExtent(x1, x2, y1, y2, z2/2, z2/2)
    # saggital.SetDisplayExtent(25, 25, 0, 49, 0, 49)
    saggital.Update()

    return saggital


def streamtube(lines, colors=None, opacity=1, linewidth=0.01, tube_sides=9,
               lod=True, lod_points=10 ** 4, lod_points_size=3,
               spline_subdiv=None, lookup_colormap=None):
    """ Uses streamtubes to visualize polylines

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), tuple (3,) or colormap
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
    dipy.viz.fvtk.line
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
    #actor.GetProperty().SetInterpolationToGouraud()
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
    lines :  list of arrays representing lines as 3d points  for example
            lines = [np.random.rand(10,3),np.random.rand(20,3)]
            represents 2 lines the first with 10 points and the second with
            20 points in x,y,z coordinates.
    colors : array, shape (N,3)
            Colormap where every triplet is encoding red, green and blue e.g.

            ::
              r1,g1,b1
              r2,g2,b2
              ...
              rN,gN,bN

            where

            ::
              0=<r<=1,
              0=<g<=1,
              0=<b<=1

    opacity : float, optional
        ``0 <= transparency <= 1``
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

    actor = vtk.vtkActor()
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
    colors : array (N, 3), tuple (3,) or colormap

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

            elif cols_arr.ndim == 1:  # the same colors for all points
                vtk_colors = numpy_to_vtk_colors(
                    np.tile(255 * cols_arr, (nb_points, 1)))

            elif cols_arr.ndim == 2:  # map color to each line
                colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
            else:  # colormap
                #  get colors for each vertex
                cols_arr = trilinear_interp(cols_arr, points_array)
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
                          saturation_range=(1, 1),  value_range=(0.8, 0.8)):
    """ Default Lookup table for the colormap
    """
    vtk_lookup_table = vtk.vtkLookupTable()
    vtk_lookup_table.SetRange(scale_range)
    vtk_lookup_table.SetTableRange(scale_range)

    vtk_lookup_table.SetHueRange(hue_range)
    vtk_lookup_table.SetSaturationRange(saturation_range)
    vtk_lookup_table.SetValueRange(value_range)

    vtk_lookup_table.Build()
    return vtk_lookup_table


def scalar_bar(lookup_table):
    """ Default Scalar bar actor for the colormap

    Deepcopy the lookup_table because sometime vtkPolyDataMapper delete it
    """
    lookup_table_copy = vtk.vtkLookupTable()
    lookup_table_copy.DeepCopy(lookup_table)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lookup_table_copy)
    scalar_bar.SetNumberOfLabels(6)

    return scalar_bar
