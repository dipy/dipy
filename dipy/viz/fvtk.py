""" Fvtk module implements simple visualization functions using VTK.

The main idea is the following:
A window can have one or more renderers. A renderer can have none,
one or more actors. Examples of actors are a sphere, line, point etc.
You basically add actors in a renderer and in that way you can
visualize the forementioned objects e.g. sphere, line ...

Examples
---------
>>> from dipy.viz import fvtk
>>> r=fvtk.ren()
>>> a=fvtk.axes()
>>> fvtk.add(r,a)
>>> #fvtk.show(r)

For more information on VTK there many neat examples in
http://www.vtk.org/Wiki/VTK/Tutorials/External_Tutorials
"""
from __future__ import division, print_function, absolute_import
from warnings import warn

from dipy.utils.six.moves import xrange

import numpy as np

from dipy.core.ndindex import ndindex

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')

cm, have_matplotlib, _ = optional_package('matplotlib.cm')

if have_matplotlib:
    get_cmap = cm.get_cmap
else:
    from dipy.data import get_cmap

from dipy.viz.colormap import create_colormap

# a track buffer used only with picking tracks
track_buffer = []
# indices buffer for the tracks
ind_buffer = []
# tempory renderer used only with picking tracks
tmp_ren = None

if have_vtk:

    major_version = vtk.vtkVersion.GetVTKMajorVersion()

    # Create a text mapper and actor to display the results of picking.
    textMapper = vtk.vtkTextMapper()
    tprop = textMapper.GetTextProperty()
    tprop.SetFontFamilyToArial()
    tprop.SetFontSize(10)
    # tprop.BoldOn()
    # tprop.ShadowOn()
    tprop.SetColor(1, 0, 0)
    textActor = vtk.vtkActor2D()
    textActor.VisibilityOff()
    textActor.SetMapper(textMapper)
    # Create a cell picker.
    picker = vtk.vtkCellPicker()

    from dipy.viz.window import (ren, renderer, add, clear, rm, rm_all,
                                 show, record, snapshot)
    from dipy.viz.actor import line, streamtube, slicer, axes

    try:
        from vtk import vtkVolumeTextureMapper2D
        have_vtk_texture_mapper2D = True
    except:
        have_vtk_texture_mapper2D = False

else:
    ren, have_ren, _ = optional_package('dipy.viz.window.ren',
                                        'Python VTK is not installed')


def dots(points, color=(1, 0, 0), opacity=1, dot_size=5):
    """ Create one or more 3d points

    Parameters
    ----------
    points : ndarray, (N, 3)
    color : tuple (3,)
    opacity : float
    dot_size : int

    Returns
    --------
    vtkActor

    See Also
    ---------
    dipy.viz.fvtk.point

    """

    if points.ndim == 2:
        points_no = points.shape[0]
    else:
        points_no = 1

    polyVertexPoints = vtk.vtkPoints()
    polyVertexPoints.SetNumberOfPoints(points_no)
    aPolyVertex = vtk.vtkPolyVertex()
    aPolyVertex.GetPointIds().SetNumberOfIds(points_no)

    cnt = 0
    if points.ndim > 1:
        for point in points:
            polyVertexPoints.InsertPoint(cnt, point[0], point[1], point[2])
            aPolyVertex.GetPointIds().SetId(cnt, cnt)
            cnt += 1
    else:
        polyVertexPoints.InsertPoint(cnt, points[0], points[1], points[2])
        aPolyVertex.GetPointIds().SetId(cnt, cnt)
        cnt += 1

    aPolyVertexGrid = vtk.vtkUnstructuredGrid()
    aPolyVertexGrid.Allocate(1, 1)
    aPolyVertexGrid.InsertNextCell(aPolyVertex.GetCellType(),
                                   aPolyVertex.GetPointIds())

    aPolyVertexGrid.SetPoints(polyVertexPoints)
    aPolyVertexMapper = vtk.vtkDataSetMapper()
    if major_version <= 5:
        aPolyVertexMapper.SetInput(aPolyVertexGrid)
    else:
        aPolyVertexMapper.SetInputData(aPolyVertexGrid)
    aPolyVertexActor = vtk.vtkActor()
    aPolyVertexActor.SetMapper(aPolyVertexMapper)

    aPolyVertexActor.GetProperty().SetColor(color)
    aPolyVertexActor.GetProperty().SetOpacity(opacity)
    aPolyVertexActor.GetProperty().SetPointSize(dot_size)
    return aPolyVertexActor


def point(points, colors, opacity=1, point_radius=0.1, theta=8, phi=8):
    """ Visualize points as sphere glyphs

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    colors : ndarray (N,3) or tuple (3,)
    point_radius : float
    theta : int
    phi : int

    Returns
    -------
    vtkActor

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> ren = fvtk.ren()
    >>> pts = np.random.rand(5, 3)
    >>> point_actor = fvtk.point(pts, fvtk.colors.coral)
    >>> fvtk.add(ren, point_actor)
    >>> #fvtk.show(ren)
    """

    if np.array(colors).ndim == 1:
        # return dots(points,colors,opacity)
        colors = np.tile(colors, (len(points), 1))

    scalars = vtk.vtkUnsignedCharArray()
    scalars.SetNumberOfComponents(3)

    pts = vtk.vtkPoints()
    cnt_colors = 0

    for p in points:

        pts.InsertNextPoint(p[0], p[1], p[2])
        scalars.InsertNextTuple3(
            round(255 * colors[cnt_colors][0]),
            round(255 * colors[cnt_colors][1]),
            round(255 * colors[cnt_colors][2]))
        cnt_colors += 1

    src = vtk.vtkSphereSource()
    src.SetRadius(point_radius)
    src.SetThetaResolution(theta)
    src.SetPhiResolution(phi)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(pts)
    polyData.GetPointData().SetScalars(scalars)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(src.GetOutputPort())
    if major_version <= 5:
        glyph.SetInput(polyData)
    else:
        glyph.SetInputData(polyData)
    glyph.SetColorModeToColorByScalar()
    glyph.SetScaleModeToDataScalingOff()
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        mapper.SetInput(glyph.GetOutput())
    else:
        mapper.SetInputData(glyph.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def label(ren, text='Origin', pos=(0, 0, 0), scale=(0.2, 0.2, 0.2),
          color=(1, 1, 1)):
    ''' Create a label actor.

    This actor will always face the camera

    Parameters
    ----------
    ren : vtkRenderer() object
       Renderer as returned by ``ren()``.
    text : str
        Text for the label.
    pos : (3,) array_like, optional
        Left down position of the label.
    scale : (3,) array_like
        Changes the size of the label.
    color : (3,) array_like
        Label color as ``(r,g,b)`` tuple.

    Returns
    -------
    l : vtkActor object
        Label.

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> l=fvtk.label(r)
    >>> fvtk.add(r,l)
    >>> #fvtk.show(r)
    '''
    atext = vtk.vtkVectorText()
    atext.SetText(text)

    textm = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        textm.SetInput(atext.GetOutput())
    else:
        textm.SetInputData(atext.GetOutput())

    texta = vtk.vtkFollower()
    texta.SetMapper(textm)
    texta.SetScale(scale)

    texta.GetProperty().SetColor(color)
    texta.SetPosition(pos)

    ren.AddActor(texta)
    texta.SetCamera(ren.GetActiveCamera())

    return texta


def volume(vol, voxsz=(1.0, 1.0, 1.0), affine=None, center_origin=1,
           info=0, maptype=0, trilinear=1, iso=0, iso_thr=100,
           opacitymap=None, colormap=None):
    ''' Create a volume and return a volumetric actor using volumetric
    rendering.

    This function has many different interesting capabilities. The maptype,
    opacitymap and colormap are the most crucial parameters here.

    Parameters
    ----------
    vol : array, shape (N, M, K), dtype uint8
        An array representing the volumetric dataset that we want to visualize
        using volumetric rendering.
    voxsz : (3,) array_like
        Voxel size.
    affine : (4, 4) ndarray
        As given by volumeimages.
    center_origin : int {0,1}
        It considers that the center of the volume is the
        point ``(-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,
            -vol.shape[2]/2.0+0.5)``.
    info : int {0,1}
        If 1 it prints out some info about the volume, the method and the
        dataset.
    trilinear : int {0,1}
        Use trilinear interpolation, default 1, gives smoother rendering. If
        you want faster interpolation use 0 (Nearest).
    maptype : int {0,1}
        The maptype is a very important parameter which affects the
        raycasting algorithm in use for the rendering.
        The options are:
        If 0 then vtkVolumeTextureMapper2D is used.
        If 1 then vtkVolumeRayCastFunction is used.
    iso : int {0,1}
        If iso is 1 and maptype is 1 then we use
        ``vtkVolumeRayCastIsosurfaceFunction`` which generates an isosurface at
        the predefined iso_thr value. If iso is 0 and maptype is 1
        ``vtkVolumeRayCastCompositeFunction`` is used.
    iso_thr : int
        If iso is 1 then then this threshold in the volume defines the value
        which will be used to create the isosurface.
    opacitymap : (2, 2) ndarray
        The opacity map assigns a transparency coefficient to every point in
        the volume.  The default value uses the histogram of the volume to
        calculate the opacitymap.
    colormap : (4, 4) ndarray
        The color map assigns a color value to every point in the volume.
        When None from the histogram it uses a red-blue colormap.

    Returns
    -------
    v : vtkVolume
        Volume.

    Notes
    --------
    What is the difference between TextureMapper2D and RayCastFunction?  Coming
    soon... See VTK user's guide [book] & The Visualization Toolkit [book] and
    VTK's online documentation & online docs.

    What is the difference between RayCastIsosurfaceFunction and
    RayCastCompositeFunction?  Coming soon... See VTK user's guide [book] &
    The Visualization Toolkit [book] and VTK's online documentation &
    online docs.

    What about trilinear interpolation?
    Coming soon... well when time permits really ... :-)

    Examples
    --------
    First example random points.

    >>> from dipy.viz import fvtk
    >>> import numpy as np
    >>> vol=100*np.random.rand(100,100,100)
    >>> vol=vol.astype('uint8')
    >>> vol.min(), vol.max()
    (0, 99)
    >>> r = fvtk.ren()
    >>> v = fvtk.volume(vol)
    >>> fvtk.add(r,v)
    >>> #fvtk.show(r)

    Second example with a more complicated function

    >>> from dipy.viz import fvtk
    >>> import numpy as np
    >>> x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    >>> s = np.sin(x*y*z)/(x*y*z)
    >>> r = fvtk.ren()
    >>> v = fvtk.volume(s)
    >>> fvtk.add(r,v)
    >>> #fvtk.show(r)

    If you find this function too complicated you can always use mayavi.
    Please do not forget to use the -wthread switch in ipython if you are
    running mayavi.

    from enthought.mayavi import mlab
    import numpy as np
    x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    s = np.sin(x*y*z)/(x*y*z)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(s))
    mlab.show()

    More mayavi demos are available here:

    http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/mlab.html

    '''
    if vol.ndim != 3:
        raise ValueError('3d numpy arrays only please')

    if info:
        print('Datatype', vol.dtype, 'converted to uint8')

    vol = np.interp(vol, [vol.min(), vol.max()], [0, 255])
    vol = vol.astype('uint8')

    if opacitymap is None:

        bin, res = np.histogram(vol.ravel())
        res2 = np.interp(res, [vol.min(), vol.max()], [0, 1])
        opacitymap = np.vstack((res, res2)).T
        opacitymap = opacitymap.astype('float32')

        '''
        opacitymap=np.array([[ 0.0, 0.0],
                          [50.0, 0.9]])
        '''

    if info:
        print('opacitymap', opacitymap)

    if colormap is None:

        bin, res = np.histogram(vol.ravel())
        res2 = np.interp(res, [vol.min(), vol.max()], [0, 1])
        zer = np.zeros(res2.shape)
        colormap = np.vstack((res, res2, zer, res2[::-1])).T
        colormap = colormap.astype('float32')

        '''
        colormap=np.array([[0.0, 0.5, 0.0, 0.0],
                                        [64.0, 1.0, 0.5, 0.5],
                                        [128.0, 0.9, 0.2, 0.3],
                                        [196.0, 0.81, 0.27, 0.1],
                                        [255.0, 0.5, 0.5, 0.5]])
        '''

    if info:
        print('colormap', colormap)

    im = vtk.vtkImageData()

    if major_version <= 5:
        im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0], vol.shape[1], vol.shape[2])
    # im.SetOrigin(0,0,0)
    # im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    if major_version <= 5:
        im.AllocateScalars()
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):

                im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

    if affine is not None:

        aff = vtk.vtkMatrix4x4()
        aff.DeepCopy((affine[0, 0], affine[0, 1], affine[0, 2],
                      affine[0, 3], affine[1, 0], affine[1, 1],
                      affine[1, 2], affine[1, 3], affine[2, 0],
                      affine[2, 1], affine[2, 2], affine[2, 3],
                      affine[3, 0], affine[3, 1], affine[3, 2],
                      affine[3, 3]))
        # aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],0,affine[1,0],affine[1,1],affine[1,2],0,affine[2,0],affine[2,1],affine[2,2],0,affine[3,0],affine[3,1],affine[3,2],1))
        # aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],127.5,affine[1,0],affine[1,1],affine[1,2],-127.5,affine[2,0],affine[2,1],affine[2,2],-127.5,affine[3,0],affine[3,1],affine[3,2],1))

        reslice = vtk.vtkImageReslice()
        if major_version <= 5:
            reslice.SetInput(im)
        else:
            reslice.SetInputData(im)
        # reslice.SetOutputDimensionality(2)
        # reslice.SetOutputOrigin(127,-145,147)

        reslice.SetResliceAxes(aff)
        # reslice.SetOutputOrigin(-127,-127,-127)
        # reslice.SetOutputExtent(-127,128,-127,128,-127,128)
        # reslice.SetResliceAxesOrigin(0,0,0)
        # print 'Get Reslice Axes Origin ', reslice.GetResliceAxesOrigin()
        # reslice.SetOutputSpacing(1.0,1.0,1.0)

        reslice.SetInterpolationModeToLinear()
        # reslice.UpdateWholeExtent()

        # print 'reslice GetOutputOrigin', reslice.GetOutputOrigin()
        # print 'reslice GetOutputExtent',reslice.GetOutputExtent()
        # print 'reslice GetOutputSpacing',reslice.GetOutputSpacing()

        changeFilter = vtk.vtkImageChangeInformation()
        if major_version <= 5:
            changeFilter.SetInput(reslice.GetOutput())
        else:
            changeFilter.SetInputData(reslice.GetOutput())
        # changeFilter.SetInput(im)
        if center_origin:
            changeFilter.SetOutputOrigin(
                -vol.shape[0] / 2.0 + 0.5,
                -vol.shape[1] / 2.0 + 0.5,
                -vol.shape[2] / 2.0 + 0.5)
            print('ChangeFilter ', changeFilter.GetOutputOrigin())

    opacity = vtk.vtkPiecewiseFunction()
    for i in range(opacitymap.shape[0]):
        opacity.AddPoint(opacitymap[i, 0], opacitymap[i, 1])

    color = vtk.vtkColorTransferFunction()
    for i in range(colormap.shape[0]):
        color.AddRGBPoint(
            colormap[i, 0], colormap[i, 1], colormap[i, 2], colormap[i, 3])

    if(maptype == 0):
        if not have_vtk_texture_mapper2D:
            raise ValueError("VolumeTextureMapper2D is not available in your "
                             "version of VTK")

        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)

        if trilinear:
            property.SetInterpolationTypeToLinear()
        else:
            property.SetInterpolationTypeToNearest()

        if info:
            print('mapper VolumeTextureMapper2D')
        mapper = vtk.vtkVolumeTextureMapper2D()
        if affine is None:
            if major_version <= 5:
                mapper.SetInput(im)
            else:
                mapper.SetInputData(im)
        else:
            if major_version <= 5:
                mapper.SetInput(changeFilter.GetOutput())
            else:
                mapper.SetInputData(changeFilter.GetOutput())

    if (maptype == 1):

        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        property.ShadeOn()
        if trilinear:
            property.SetInterpolationTypeToLinear()
        else:
            property.SetInterpolationTypeToNearest()

        if iso:
            isofunc = vtk.vtkVolumeRayCastIsosurfaceFunction()
            isofunc.SetIsoValue(iso_thr)
        else:
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

        if info:
            print('mapper VolumeRayCastMapper')

        mapper = vtk.vtkVolumeRayCastMapper()
        if iso:
            mapper.SetVolumeRayCastFunction(isofunc)
            if info:
                print('Isosurface')
        else:
            mapper.SetVolumeRayCastFunction(compositeFunction)

            # mapper.SetMinimumImageSampleDistance(0.2)
            if info:
                print('Composite')

        if affine is None:
            if major_version <= 5:
                mapper.SetInput(im)
            else:
                mapper.SetInputData(im)
        else:
            # mapper.SetInput(reslice.GetOutput())
            if major_version <= 5:
                mapper.SetInput(changeFilter.GetOutput())
            else:
                mapper.SetInputData(changeFilter.GetOutput())
            # Return mid position in world space
            # im2=reslice.GetOutput()
            # index=im2.FindPoint(vol.shape[0]/2.0,vol.shape[1]/2.0,vol.shape[2]/2.0)
            # print 'Image Getpoint ' , im2.GetPoint(index)

    volum = vtk.vtkVolume()
    volum.SetMapper(mapper)
    volum.SetProperty(property)

    if info:

        print('Origin', volum.GetOrigin())
        print('Orientation', volum.GetOrientation())
        print('OrientationW', volum.GetOrientationWXYZ())
        print('Position', volum.GetPosition())
        print('Center', volum.GetCenter())
        print('Get XRange', volum.GetXRange())
        print('Get YRange', volum.GetYRange())
        print('Get ZRange', volum.GetZRange())
        print('Volume data type', vol.dtype)

    return volum


def contour(vol, voxsz=(1.0, 1.0, 1.0), affine=None, levels=[50],
            colors=[np.array([1.0, 0.0, 0.0])], opacities=[0.5]):
    """ Take a volume and draw surface contours for any any number of
    thresholds (levels) where every contour has its own color and opacity

    Parameters
    ----------
    vol : (N, M, K) ndarray
        An array representing the volumetric dataset for which we will draw
        some beautiful contours .
    voxsz : (3,) array_like
        Voxel size.
    affine : None
        Not used.
    levels : array_like
        Sequence of thresholds for the contours taken from image values needs
        to be same datatype as `vol`.
    colors : (N, 3) ndarray
        RGB values in [0,1].
    opacities : array_like
        Opacities of contours.

    Returns
    -------
    vtkAssembly

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.viz import fvtk
    >>> A=np.zeros((10,10,10))
    >>> A[3:-3,3:-3,3:-3]=1
    >>> r=fvtk.ren()
    >>> fvtk.add(r,fvtk.contour(A,levels=[1]))
    >>> #fvtk.show(r)

    """

    im = vtk.vtkImageData()
    if major_version <= 5:
        im.SetScalarTypeToUnsignedChar()

    im.SetDimensions(vol.shape[0], vol.shape[1], vol.shape[2])
    # im.SetOrigin(0,0,0)
    # im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    if major_version <= 5:
        im.AllocateScalars()
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):

                im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

    ass = vtk.vtkAssembly()
    # ass=[]

    for (i, l) in enumerate(levels):

        # print levels
        skinExtractor = vtk.vtkContourFilter()
        if major_version <= 5:
            skinExtractor.SetInput(im)
        else:
            skinExtractor.SetInputData(im)
        skinExtractor.SetValue(0, l)

        skinNormals = vtk.vtkPolyDataNormals()
        skinNormals.SetInputConnection(skinExtractor.GetOutputPort())
        skinNormals.SetFeatureAngle(60.0)

        skinMapper = vtk.vtkPolyDataMapper()
        skinMapper.SetInputConnection(skinNormals.GetOutputPort())
        skinMapper.ScalarVisibilityOff()

        skin = vtk.vtkActor()

        skin.SetMapper(skinMapper)
        skin.GetProperty().SetOpacity(opacities[i])

        # print colors[i]
        skin.GetProperty().SetColor(colors[i][0], colors[i][1], colors[i][2])
        # skin.Update()
        ass.AddPart(skin)

        del skin
        del skinMapper
        del skinExtractor

    return ass


def _makeNd(array, ndim):
    """Pads as many 1s at the beginning of array's shape as are need to give
    array ndim dimensions."""
    new_shape = (1,) * (ndim - array.ndim) + array.shape
    return array.reshape(new_shape)


def sphere_funcs(sphere_values, sphere, image=None, colormap='jet',
                 scale=2.2, norm=True, radial_scale=True):
    """Plot many morphed spherical functions simultaneously.

    Parameters
    ----------
    sphere_values : (M,) or (X, M) or (X, Y, M) or (X, Y, Z, M) ndarray
        Values on the sphere.
    sphere : Sphere
    image : None,
        Not  yet supported.
    colormap : None or 'jet'
        If None then no color is used.
    scale : float,
        Distance between spheres.
    norm : bool,
        Normalize `sphere_values`.
    radial_scale : bool,
        Scale sphere points according to odf values.

    Returns
    -------
    actor : vtkActor
        Spheres.

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> r = fvtk.ren()
    >>> odfs = np.ones((5, 5, 724))
    >>> odfs[..., 0] = 2.
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric724')
    >>> fvtk.add(r, fvtk.sphere_funcs(odfs, sphere))
    >>> #fvtk.show(r)

    """

    sphere_values = np.asarray(sphere_values)
    if sphere_values.ndim > 4:
        raise ValueError("Wrong shape")
    sphere_values = _makeNd(sphere_values, 4)

    grid_shape = np.array(sphere_values.shape[:3])
    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    if sphere_values.shape[-1] != sphere.vertices.shape[0]:
        msg = 'Sphere.vertices.shape[0] should be the same as the '
        msg += 'last dimensions of sphere_values i.e. sphere_values.shape[-1]'
        raise ValueError(msg)

    list_sq = []
    list_cols = []

    for ijk in np.ndindex(*grid_shape):
        m = sphere_values[ijk].copy()

        if norm:
            m /= abs(m).max()

        if radial_scale:
            xyz = vertices.T * m
        else:
            xyz = vertices.T.copy()

        xyz += scale * (ijk - grid_shape / 2.)[:, None]

        xyz = xyz.T

        list_sq.append(xyz)
        if colormap is not None:
            cols = create_colormap(m, colormap)
            cols = np.interp(cols, [0, 1], [0, 255]).astype('ubyte')
            list_cols.append(cols)

    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()
    if colormap is not None:
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

    for k in xrange(len(list_sq)):

        xyz = list_sq[k]
        if colormap is not None:
            cols = list_cols[k]

        for i in xrange(xyz.shape[0]):

            points.InsertNextPoint(*xyz[i])
            if colormap is not None:
                colors.InsertNextTuple3(*cols[i])

        for j in xrange(faces.shape[0]):

            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, faces[j, 0] + k * xyz.shape[0])
            triangle.GetPointIds().SetId(1, faces[j, 1] + k * xyz.shape[0])
            triangle.GetPointIds().SetId(2, faces[j, 2] + k * xyz.shape[0])
            triangles.InsertNextCell(triangle)
            del triangle

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    if colormap is not None:
        polydata.GetPointData().SetScalars(colors)
    polydata.Modified()

    mapper = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def peaks(peaks_dirs, peaks_values=None, scale=2.2, colors=(1, 0, 0)):
    """ Visualize peak directions as given from ``peaks_from_model``

    Parameters
    ----------
    peaks_dirs : ndarray
        Peak directions. The shape of the array can be (M, 3) or (X, M, 3) or
        (X, Y, M, 3) or (X, Y, Z, M, 3)
    peaks_values : ndarray
        Peak values. The shape of the array can be (M, ) or (X, M) or
        (X, Y, M) or (X, Y, Z, M)

    scale : float
        Distance between spheres

    colors : ndarray or tuple
        Peak colors

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

    list_dirs = []

    for ijk in np.ndindex(*grid_shape):

        xyz = scale * (ijk - grid_shape / 2.)[:, None]

        xyz = xyz.T

        for i in range(peaks_dirs.shape[-2]):

            if peaks_values is not None:

                pv = peaks_values[ijk][i]

            else:

                pv = 1.

            symm = np.vstack((-peaks_dirs[ijk][i] * pv + xyz,
                              peaks_dirs[ijk][i] * pv + xyz))

            list_dirs.append(symm)

    return line(list_dirs, colors)


def tensor(evals, evecs, scalar_colors=None,
           sphere=None, scale=2.2, norm=True):
    """Plot many tensors as ellipsoids simultaneously.

    Parameters
    ----------
    evals : (3,) or (X, 3) or (X, Y, 3) or (X, Y, Z, 3) ndarray
        eigenvalues
    evecs : (3, 3) or (X, 3, 3) or (X, Y, 3, 3) or (X, Y, Z, 3, 3) ndarray
        eigenvectors
    scalar_colors : (3,) or (X, 3) or (X, Y, 3) or (X, Y, Z, 3) ndarray
        RGB colors used to show the tensors
        Default None, color the ellipsoids using ``color_fa``
    sphere : Sphere,
        this sphere will be transformed to the tensor ellipsoid
        Default is None which uses a symmetric sphere with 724 points.
    scale : float,
        distance between ellipsoids.
    norm : boolean,
        Normalize `evals`.

    Returns
    -------
    actor : vtkActor
        Ellipsoids

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> r = fvtk.ren()
    >>> evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    >>> evecs = np.eye(3)
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric724')
    >>> fvtk.add(r, fvtk.tensor(evals, evecs, sphere=sphere))
    >>> #fvtk.show(r)

    """

    evals = np.asarray(evals)
    if evals.ndim > 4:
        raise ValueError("Wrong shape")
    evals = _makeNd(evals, 4)
    evecs = _makeNd(evecs, 5)

    grid_shape = np.array(evals.shape[:3])

    if sphere is None:
        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    if scalar_colors is None:
        from dipy.reconst.dti import color_fa, fractional_anisotropy
        cfa = color_fa(fractional_anisotropy(evals), evecs)
    else:
        cfa = _makeNd(scalar_colors, 4)

    list_sq = []
    list_cols = []

    for ijk in ndindex(grid_shape):
        ea = evals[ijk]
        if norm:
            ea /= ea.max()
        ea = np.diag(ea.copy())

        ev = evecs[ijk].copy()
        xyz = np.dot(ev, np.dot(ea, vertices.T))

        xyz += scale * (ijk - grid_shape / 2.)[:, None]

        xyz = xyz.T

        list_sq.append(xyz)

        acolor = np.zeros(xyz.shape)
        acolor[:, :] = np.interp(cfa[ijk], [0, 1], [0, 255])
        list_cols.append(acolor.astype('ubyte'))

    points = vtk.vtkPoints()
    triangles = vtk.vtkCellArray()

    for k in xrange(len(list_sq)):

        xyz = list_sq[k]

        cols = list_cols[k]

        for i in xrange(xyz.shape[0]):

            points.InsertNextPoint(*xyz[i])
            colors.InsertNextTuple3(*cols[i])

        for j in xrange(faces.shape[0]):

            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, faces[j, 0] + k * xyz.shape[0])
            triangle.GetPointIds().SetId(1, faces[j, 1] + k * xyz.shape[0])
            triangle.GetPointIds().SetId(2, faces[j, 2] + k * xyz.shape[0])
            triangles.InsertNextCell(triangle)
            del triangle

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    polydata.GetPointData().SetScalars(colors)
    polydata.Modified()

    mapper = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def camera(ren, pos=None, focal=None, viewup=None, verbose=True):
    """ Change the active camera

    Parameters
    ----------
    ren : vtkRenderer
    pos : tuple
        (x, y, z) position of the camera
    focal : tuple
        (x, y, z) focal point
    viewup : tuple
        (x, y, z) viewup vector
    verbose : bool
        show information about the camera

    Returns
    -------
    vtkCamera
    """

    msg = "This function is deprecated."
    msg += "Please use the window.Renderer class to get/set the active camera."
    warn(DeprecationWarning(msg))

    cam = ren.GetActiveCamera()
    if verbose:
        print('Camera Position (%.2f,%.2f,%.2f)' % cam.GetPosition())
        print('Camera Focal Point (%.2f,%.2f,%.2f)' % cam.GetFocalPoint())
        print('Camera View Up (%.2f,%.2f,%.2f)' % cam.GetViewUp())
    if pos is not None:
        cam = ren.GetActiveCamera().SetPosition(*pos)
    if focal is not None:
        ren.GetActiveCamera().SetFocalPoint(*focal)
    if viewup is not None:
        ren.GetActiveCamera().SetViewUp(*viewup)

    cam = ren.GetActiveCamera()
    if pos is not None or focal is not None or viewup is not None:
        if verbose:
            print('-------------------------------------')
            print('Camera New Position (%.2f,%.2f,%.2f)' % cam.GetPosition())
            print('Camera New Focal Point (%.2f,%.2f,%.2f)' %
                  cam.GetFocalPoint())
            print('Camera New View Up (%.2f,%.2f,%.2f)' % cam.GetViewUp())

    return cam


if __name__ == "__main__":
    pass
