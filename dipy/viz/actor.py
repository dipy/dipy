from __future__ import division, print_function, absolute_import

import numpy as np
from nibabel.affines import apply_affine

<<<<<<< HEAD
from dipy.viz.colormap import colormap_lookup_table, create_colormap
from dipy.viz.utils import lines_to_vtk_polydata
from dipy.viz.utils import set_input
=======
from dipy.viz.colormap import line_colors
from dipy.viz.utils import numpy_to_vtk_points, numpy_to_vtk_colors
from dipy.viz.utils import vtk_matrix_to_numpy, numpy_to_vtk_matrix
from dipy.viz.utils import set_input, map_coordinates_3d_4d, get_bounding_box_sizes, shallow_copy
from dipy.viz import layout
from dipy.core.ndindex import ndindex
>>>>>>> 673537700ce0828891541d053481f728b7ed5253

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
<<<<<<< HEAD
           lookup_colormap=None, interpolation='linear'):
    """ Cuts 3D scalar or rgb volumes into 2D images
=======
           lookup_colormap=None):
    """ Cuts 3D scalar or rgb volumes into images
>>>>>>> 673537700ce0828891541d053481f728b7ed5253

    Parameters
    ----------
    data : array, shape (X, Y, Z) or (X, Y, Z, 3)
        A grayscale or rgb 4D volume as a numpy array.
    affine : array, shape (4, 4)
        Grid to space (usually RAS 1mm) transformation matrix. Default is None.
        If None then the identity matrix is used.
    value_range : None or tuple (2,)
        If None then the values will be interpolated from (data.min(),
        data.max()) to (0, 255). Otherwise from (value_range[0],
        value_range[1]) to (0, 255).
    opacity : float
        Opacity of 0 means completely transparent and 1 completely visible.
    lookup_colormap : vtkLookupTable
        If None (default) then a grayscale map is created.
    interpolation : string
        If 'linear' (default) then linear interpolation is used on the final
        texture mapping. If 'nearest' then nearest neighbor interpolation is
        used on the final texture mapping.

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
    voxsz = (1.0, 1.0, 1.0)
    # im.SetOrigin(0,0,0)

    im.SetSpacing(*voxsz)#voxsz[2], voxsz[0], voxsz[1])
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
    # image_resliced.SetInformationInput(im)
    image_resliced.SetResliceTransform(transform)

    image_resliced.AutoCropOutputOn()

<<<<<<< HEAD
    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates

    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    image_resliced.SetOutputSpacing(*zooms)

    image_resliced.SetInterpolationModeToLinear()
=======
    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    image_resliced.SetOutputSpacing(*zooms)
    #image_resliced.SetOutputExtent(0, 6, 0, 12, 0, 3*7+1)

    #image_resliced.SetInterpolationModeToCubic()
    image_resliced.SetInterpolationModeToNearestNeighbor()
>>>>>>> 673537700ce0828891541d053481f728b7ed5253
    image_resliced.Update()

    if nb_components == 1:
        if lookup_colormap is None:
            # Create a black/white lookup table.
            lut = colormap_lookup_table((0, 255), (0, 0), (0, 0), (0, 1))
        else:
            lut = lookup_colormap

    x1, x2, y1, y2, z1, z2 = im.GetExtent()
    print('Hey')
    print(im.GetExtent())

    ex1, ex2, ey1, ey2, ez1, ez2 = image_resliced.GetOutput().GetExtent()
    print('Yo')
    print(image_resliced.GetOutput().GetExtent())

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
                self.display_extent(ex1, ex2, ey1, ey2, ez2//2, ez2//2)
            if x is not None:
                x = min(max(0, x), self.shape[0])
                self.display_extent(x, x, ey1, ey2, ez1, ez2)
            if y is not None:
                y = min(max(0, y), self.shape[1])
                self.display_extent(ex1, ex2, y, y, ez1, ez2)
            if z is not None:
                z = min(max(0, z), self.shape[2])
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
            if interpolation == 'nearest':
                im_actor.SetInterpolate(False)
            else:
                im_actor.SetInterpolate(True)
            if major_version >= 6:
                im_actor.GetMapper().BorderOn()
            return im_actor

<<<<<<< HEAD
=======
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

>>>>>>> 673537700ce0828891541d053481f728b7ed5253
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
    if major_version >= 6:
        image_actor.GetMapper().BorderOn()

    if interpolation == 'nearest':
        image_actor.SetInterpolate(False)
    else:
        image_actor.SetInterpolate(True)

    if major_version >= 6:
        image_actor.GetMapper().BorderOn()

    return image_actor


<<<<<<< HEAD
def streamtube(lines, colors=None, opacity=1, linewidth=0.1, tube_sides=9,
=======
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
>>>>>>> 673537700ce0828891541d053481f728b7ed5253
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
        Default is 1.
    linewidth : float
        Default is 0.01.
    tube_sides : int
        Default is 9.
    lod : bool
        Use vtkLODActor(level of detail) rather than vtkActor. Default is True.
        Level of detail actors do not render the full geometry when the
        frame rate is low.
    lod_points : int
        Number of points to be used when LOD is in effect. Default is 10000.
    lod_points_size : int
        Size of points when lod is in effect. Default is 3.
    spline_subdiv : int
        Number of splines subdivision to smooth streamtubes. Default is None.
    lookup_colormap : vtkLookupTable
        Add a default lookup table to the colormap. Default is None which calls
        :func:`dipy.viz.actor.colormap_lookup_table`.

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.viz import actor, window
    >>> ren = window.Renderer()
    >>> lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    >>> colors = np.random.rand(2, 3)
    >>> c = actor.streamtube(lines, colors)
    >>> ren.add(c)
    >>> #window.show(ren)

    Notes
    -----
    Streamtubes can be heavy on GPU when loading many streamlines and
    therefore, you may experience slow rendering time depending on system GPU.
    A solution to this problem is to reduce the number of points in each
    streamline. In Dipy we provide an algorithm that will reduce the number of
    points on the straighter parts of the streamline but keep more points on
    the curvier parts. This can be used in the following way::

        from dipy.tracking.distances import approx_polygon_track
        lines = [approx_polygon_track(line, 0.2) for line in lines]

    Alternatively we suggest using the ``line`` actor which is much more
    efficient.

    See Also
    --------
    :func:`dipy.viz.actor.line`
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
    # TODO using the line above we will be able to visualize
    # streamtubes of varying radius
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
    actor.GetProperty().SetAmbient(0.1)
    actor.GetProperty().SetDiffuse(0.15)
    actor.GetProperty().SetSpecular(0.05)
    actor.GetProperty().SetSpecularPower(6)
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
        Default is 1.

    linewidth : float, optional
        Line thickness. Default is 1.
    spline_subdiv : int, optional
        Number of splines subdivision to smooth streamtubes. Default is None
        which means no subdivision.
    lod : bool
        Use vtkLODActor(level of detail) rather than vtkActor. Default is True.
        Level of detail actors do not render the full geometry when the
        frame rate is low.
    lod_points : int
        Number of points to be used when LOD is in effect. Default is 10000.
    lod_points_size : int
        Size of points when lod is in effect. Default is 3.
    lookup_colormap : bool, optional
        Add a default lookup table to the colormap. Default is None which calls
        :func:`dipy.viz.actor.colormap_lookup_table`.

    Returns
    ----------
    v : vtkActor or vtkLODActor object
        Line.

    Examples
    ----------
    >>> from dipy.viz import actor, window
    >>> ren = window.Renderer()
    >>> lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    >>> colors = np.random.rand(2, 3)
    >>> c = actor.line(lines, colors)
    >>> ren.add(c)
    >>> #window.show(ren)
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


def scalar_bar(lookup_table=None, title=" "):
    """ Default scalar bar actor for a given colormap (colorbar)

    Parameters
    ----------
    lookup_table : vtkLookupTable or None
        If None then ``colormap_lookup_table`` is called with default options.
    title : str

    Returns
    -------
    scalar_bar : vtkScalarBarActor

    See Also
    --------
    :func:`dipy.viz.actor.colormap_lookup_table`

    """
    lookup_table_copy = vtk.vtkLookupTable()
    if lookup_table is None:
        lookup_table = colormap_lookup_table()
    # Deepcopy the lookup_table because sometimes vtkPolyDataMapper deletes it
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
        Axes size e.g. (100, 100, 100). Default is (1, 1, 1).
    colorx : tuple (3,)
        x-axis color. Default red (1, 0, 0).
    colory : tuple (3,)
        y-axis color. Default green (0, 1, 0).
    colorz : tuple (3,)
        z-axis color. Default blue (0, 0, 1).

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


<<<<<<< HEAD
def odf_slicer(odfs, affine=None, mask=None, sphere=None, scale=2.2,
               norm=True, radial_scale=True, opacity=1.,
               colormap='plasma', global_cm=False):
    """ Slice spherical fields in native or world coordinates

    Parameters
    ----------
    odfs : ndarray
        4D array of spherical functions
    affine : array
        4x4 transformation array from native coordinates to world coordinates
    mask : ndarray
        3D mask
    sphere : Sphere
        a sphere
    scale : float
        Distance between spheres.
    norm : bool
        Normalize `sphere_values`.
    radial_scale : bool
        Scale sphere points according to odf values.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque)
    colormap : None or str
        If None then white color is used. Otherwise the name of colormap is
        given. Matplotlib colormaps are supported (e.g., 'inferno').
    global_cm : bool
        If True the colormap will be applied in all ODFs. If False
        it will be applied individually at each voxel (default False).
    """

    if mask is None:
        mask = np.ones(odfs.shape[:3], dtype=np.bool)
    else:
        mask = mask.astype(np.bool)

    szx, szy, szz = odfs.shape[:3]

    class OdfSlicerActor(vtk.vtkLODActor):

        def display_extent(self, x1, x2, y1, y2, z1, z2):
            tmp_mask = np.zeros(odfs.shape[:3], dtype=np.bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            self.mapper = _odf_slicer_mapper(odfs=odfs,
                                             affine=affine,
                                             mask=tmp_mask,
                                             sphere=sphere,
                                             scale=scale,
                                             norm=norm,
                                             radial_scale=radial_scale,
                                             opacity=opacity,
                                             colormap=colormap,
                                             global_cm=global_cm)
            self.SetMapper(self.mapper)

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1,
                                    int(np.floor(szz/2)), int(np.floor(szz/2)))
            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    odf_actor = OdfSlicerActor()
    odf_actor.display_extent(0, szx - 1, 0, szy - 1,
                             int(np.floor(szz/2)), int(np.floor(szz/2)))

    return odf_actor


def _odf_slicer_mapper(odfs, affine=None, mask=None, sphere=None, scale=2.2,
                       norm=True, radial_scale=True, opacity=1.,
                       colormap='plasma', global_cm=False):
    """ Helper function for slicing spherical fields

    Parameters
    ----------
    odfs : ndarray
        4D array of spherical functions
    affine : array
        4x4 transformation array from native coordinates to world coordinates
    mask : ndarray
        3D mask
    sphere : Sphere
        a sphere
    scale : float
        Distance between spheres.
    norm : bool
        Normalize `sphere_values`.
    radial_scale : bool
        Scale sphere points according to odf values.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque)
    colormap : None or str
        If None then white color is used. Otherwise the name of colormap is
        given. Matplotlib colormaps are supported (e.g., 'inferno').
    global_cm : bool
        If True the colormap will be applied in all ODFs. If False
        it will be applied individually at each voxel (default False).
    """
    if mask is None:
        mask = np.ones(odfs.shape[:3])

    ijk = np.ascontiguousarray(np.array(np.nonzero(mask)).T)

    if len(ijk) == 0:
        return None

    if affine is not None:
        ijk = np.ascontiguousarray(apply_affine(affine, ijk))

    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    all_xyz = []
    all_faces = []
    all_ms = []
    for (k, center) in enumerate(ijk):

        m = odfs[tuple(center.astype(np.int))].copy()

        if norm:
            m /= np.abs(m).max()

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
    if global_cm:
        all_ms = np.ascontiguousarray(
            np.concatenate(all_ms), dtype='f4')

    points = vtk.vtkPoints()
    points.SetData(all_xyz_vtk)

    cells = vtk.vtkCellArray()
    cells.SetCells(ncells, all_faces_vtk)

    if colormap is not None:
        if global_cm:
            cols = create_colormap(all_ms.ravel(), colormap)
        else:
            cols = np.zeros((ijk.shape[0],) + sphere.vertices.shape,
                            dtype='f4')
            for k in range(ijk.shape[0]):
                tmp = create_colormap(all_ms[k].ravel(), colormap)
                cols[k] = tmp.copy()

            cols = np.ascontiguousarray(
                np.reshape(cols, (cols.shape[0] * cols.shape[1],
                           cols.shape[2])), dtype='f4')

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

    return mapper


def _makeNd(array, ndim):
    """Pads as many 1s at the beginning of array's shape as are need to give
    array ndim dimensions."""
    new_shape = (1,) * (ndim - array.ndim) + array.shape
    return array.reshape(new_shape)


def peak_slicer(peaks_dirs, peaks_values=None, mask=None, affine=None,
                colors=(1, 0, 0), opacity=1, linewidth=1,
                lod=False, lod_points=10 ** 4, lod_points_size=3):
    """ Visualize peak directions as given from ``peaks_from_model``

    Parameters
    ----------
    peaks_dirs : ndarray
        Peak directions. The shape of the array can be (M, 3) or (X, M, 3) or
        (X, Y, M, 3) or (X, Y, Z, M, 3)
    peaks_values : ndarray
        Peak values. The shape of the array can be (M, ) or (X, M) or
        (X, Y, M) or (X, Y, Z, M)

    colors : tuple or None
        Default red color. If None then every peak gets an orientation color
        in similarity to a DEC map.

    opacity : float, optional
        Default is 1.

    linewidth : float, optional
        Line thickness. Default is 1.

    lod : bool
        Use vtkLODActor(level of detail) rather than vtkActor.
        Default is False. Level of detail actors do not render the full
        geometry when the frame rate is low.
    lod_points : int
        Number of points to be used when LOD is in effect. Default is 10000.
    lod_points_size : int
        Size of points when lod is in effect. Default is 3.

    Returns
    -------
    vtkActor

    See Also
    --------
    dipy.viz.fvtk.sphere_funcs

    """
    peaks_dirs = np.asarray(peaks_dirs)
    if peaks_dirs.ndim > 5:
        raise ValueError("Wrong shape")

    peaks_dirs = _makeNd(peaks_dirs, 5)
    if peaks_values is not None:
        peaks_values = _makeNd(peaks_values, 4)

    grid_shape = np.array(peaks_dirs.shape[:3])

    if mask is None:
        mask = np.ones(grid_shape).astype(np.bool)

    class PeakSlicerActor(vtk.vtkLODActor):

        def display_extent(self, x1, x2, y1, y2, z1, z2):

            tmp_mask = np.zeros(grid_shape, dtype=np.bool)
            tmp_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = True
            tmp_mask = np.bitwise_and(tmp_mask, mask)

            ijk = np.ascontiguousarray(np.array(np.nonzero(tmp_mask)).T)
            if len(ijk) == 0:
                self.SetMapper(None)
                return
            if affine is not None:
                ijk_trans = np.ascontiguousarray(apply_affine(affine, ijk))
            list_dirs = []
            for index, center in enumerate(ijk):
                # center = tuple(center)
                if affine is None:
                    xyz = center[:, None]
                else:
                    xyz = ijk_trans[index][:, None]
                xyz = xyz.T
                for i in range(peaks_dirs[tuple(center)].shape[-2]):

                    if peaks_values is not None:
                        pv = peaks_values[tuple(center)][i]
                    else:
                        pv = 1.
                    symm = np.vstack((-peaks_dirs[tuple(center)][i] * pv + xyz,
                                      peaks_dirs[tuple(center)][i] * pv + xyz))
                    list_dirs.append(symm)

            self.mapper = line(list_dirs, colors=colors,
                               opacity=opacity, linewidth=linewidth,
                               lod=lod, lod_points=lod_points,
                               lod_points_size=lod_points_size).GetMapper()
            self.SetMapper(self.mapper)

        def display(self, x=None, y=None, z=None):
            if x is None and y is None and z is None:
                self.display_extent(0, szx - 1, 0, szy - 1,
                                    int(np.floor(szz/2)), int(np.floor(szz/2)))
            if x is not None:
                self.display_extent(x, x, 0, szy - 1, 0, szz - 1)
            if y is not None:
                self.display_extent(0, szx - 1, y, y, 0, szz - 1)
            if z is not None:
                self.display_extent(0, szx - 1, 0, szy - 1, z, z)

    peak_actor = PeakSlicerActor()

    szx, szy, szz = grid_shape
    peak_actor.display_extent(0, szx - 1, 0, szy - 1,
                              int(np.floor(szz / 2)), int(np.floor(szz / 2)))

    return peak_actor
=======
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


class Container(object):
    """ Provides functionalities for grouping multiple actors using a given
    layout.

    Attributes
    ----------
    anchor : 3-tuple of float
        Anchor of this container used when laying out items in a container.
        The anchor point is relative to the center of the container.
        Default: (0, 0, 0).

    padding : 6-tuple of float
        Padding around this container bounding box. The 6-tuple represents
        (pad_x_neg, pad_x_pos, pad_y_neg, pad_y_pos, pad_z_neg, pad_z_pos).
        Default: (0, 0, 0, 0, 0, 0)

    """
    def __init__(self, layout=layout.Layout()):
        """
        Parameters
        ----------
        layout : ``dipy.viz.layout.Layout`` object
            Items of this container will be arranged according to `layout`.
        """
        self.layout = layout
        self._items = []
        self._need_update = True
        self._position = np.zeros(3)
        self._visibility = True
        self.anchor = np.zeros(3)
        self.padding = np.zeros(6)

    @property
    def items(self):
        if self._need_update:
            self.update()

        return self._items

    def add(self, *items, **kwargs):
        """ Adds some items to this container.

        Parameters
        ----------
        items : `vtkProp3D` objects
            Items to add to this container.
        borrow : bool
            If True the items are added as-is, otherwise a shallow copy is
            made first. If you intend to reuse the items elsewhere you
            should set `borrow=False`. Default: True.
        """
        self._need_update = True

        for item in items:
            if not kwargs.get('borrow', True):
                item = shallow_copy(item)

            self._items.append(item)

    def clear(self):
        """ Clears all items of this container. """
        self._need_update = True
        del self._items[:]

    def update(self):
        """ Updates the position of the items of this container. """
        self.layout.apply(self._items)
        self._need_update = False

    def add_to_renderer(self, ren):
        """ Adds the items of this container to a given renderer. """
        for item in self.items:
            if isinstance(item, Container):
                item.add_to_renderer(ren)
            else:
                ren.add(item)

    def GetBounds(self):
        """ Get the bounds of the container. """
        bounds = np.zeros(6)    # x1, x2, y1, y2, z1, z2
        bounds[::2] = np.inf    # x1, y1, z1
        bounds[1::2] = -np.inf  # x2, y2, z2

        for item in self.items:
            item_bounds = item.GetBounds()
            bounds[::2] = np.minimum(bounds[::2], item_bounds[::2])
            bounds[1::2] = np.maximum(bounds[1::2], item_bounds[1::2])

        # Add padding, if any.
        bounds[::2] -= self.padding[::2]
        bounds[1::2] += self.padding[1::2]

        return tuple(bounds)

    def GetVisibility(self):
        return self._visibility

    def SetVisibility(self, visibility):
        self._visibility = visibility
        for item in self.items:
            item.SetVisibility(visibility)

    def GetPosition(self):
        return self._position

    def AddPosition(self, position):
        self._position += position
        for item in self.items:
            item.AddPosition(position)

    def SetPosition(self, position):
        self.AddPosition(np.array(position) - self._position)

    def GetCenter(self):
        """ Get the center of the bounding box. """
        x1, x2, y1, y2, z1, z2 = self.GetBounds()
        return ((x1+x2)/2., (y1+y2)/2., (z1+z2)/2.)

    def GetLength(self):
        """ Get the length of bounding box diagonal. """
        x1, x2, y1, y2, z1, z2 = self.GetBounds()
        width, height, depth = x2-x1, y2-y1, z2-z1
        return np.sqrt(np.sum([width**2, height**2, depth**2]))

    def NewInstance(self):
        return Container(layout=self.layout)

    def ShallowCopy(self, other):
        self._position = other._position.copy()
        self._anchor = other._anchor
        self.clear()
        self.add(*other._items, borrow=False)
        self.update()

    def __len__(self):
        return len(self._items)


def grid(actors, captions=None, caption_offset=(0, -100, 0), cell_padding=0, cell_shape="rect", aspect_ratio=16/9., dim=None):
    """ Creates a grid of actors that lies in the xy-plane.

        Parameters
        ----------
        actors : list of `vtkProp3D` objects
            Actors to be layout in a grid manner.
        captions : list of `vtkProp3D` objects (optional)
            Objects serving as captions (can be any `vtkProp3D` object, not
            necessarily text). There should be one caption per actor. By
            default, there are no captions.
        caption_offset : 3-tuple of float (optional)
            Tells where to position the caption w.r.t. the center of its
            associated actor. Default: (0, -100, 0).
        cell_padding : 2-tuple of float or float (optional)
            Each grid cell will be padded according to (pad_x, pad_y) i.e.
            horizontally and vertically. Padding is evenly distributed on each
            side of the cell. If a single float is provided then both pad_x and
            pad_y will have the same value.
        cell_shape : {'rect', 'square', 'diagonal'} (optional)
            Specifies the desired shape of every grid cell.
            'rect' ensures the cells are the tightest.
            'square' ensures the cells are as wide as high.
            'diagonal' ensures the content of the cells can be rotated without
                       colliding with content of the neighboring cells.
        aspect_ratio : float (optional)
            Aspect ratio of the grid (width/height). Default: 16:9.
        dim : tuple of int (optional)
            Dimension (nb_rows, nb_cols) of the grid. If provided,
            `aspect_ratio` will be ignored.

        Returns
        -------
        ``dipy.viz.actor.Container`` object
            Object that represents the grid containing all the actors and
            captions, if any.
    """
    grid_layout = layout.GridLayout(cell_padding=cell_padding, cell_shape=cell_shape, aspect_ratio=aspect_ratio, dim=dim)
    grid = Container(layout=grid_layout)

    if captions is not None:
        actors_with_caption = []
        for actor, caption in zip(actors, captions):
            actor_center = np.array(actor.GetCenter())

            # Offset accordingly the caption w.r.t. the center of the associated actor.
            caption = shallow_copy(caption)
            caption.SetPosition(actor_center + caption_offset)

            actor_with_caption = Container()
            actor_with_caption.add(actor, caption)

            # We change the anchor of the container so the actor will be centered in the grid cell.
            actor_with_caption.anchor = actor_center - actor_with_caption.GetCenter()
            actors_with_caption.append(actor_with_caption)

        actors = actors_with_caption

    grid.add(*actors)
    return grid
>>>>>>> 673537700ce0828891541d053481f728b7ed5253
