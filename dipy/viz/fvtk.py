''' Fvtk module implements simple visualization functions using VTK.

The main idea is the following:
A window can have one or more renderers. A renderer can have none, one or more actors. Examples of actors are a sphere, line, point etc.
You basically add actors in a renderer and in that way you can visualize the forementioned objects e.g. sphere, line ...

Examples
---------
>>> from dipy.viz import fvtk
>>> r=fvtk.ren()
>>> a=fvtk.axes()
>>> fvtk.add(r,a)
>>> #fvtk.show(r)
'''
from __future__ import division, print_function, absolute_import

from dipy.utils.six.moves import xrange

import types

import numpy as np

import scipy as sp

from dipy.core.ndindex import ndindex

# Conditional import machinery for vtk
from ..utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')

'''
For more color names see
http://www.colourlovers.com/blog/2007/07/24/32-common-color-names-for-easy-reference/
'''
# Some common colors
red = np.array([1, 0, 0])
green = np.array([0, 1, 0])
blue = np.array([0, 0, 1])
yellow = np.array([1, 1, 0])
cyan = np.array([0, 1, 1])
azure = np.array([0, 0.49, 1])
golden = np.array([1, 0.84, 0])
white = np.array([1, 1, 1])
black = np.array([0, 0, 0])

aquamarine = np.array([0.498, 1., 0.83])
indigo = np.array([0.29411765, 0., 0.50980392])
lime = np.array([0.74901961, 1., 0.])
hot_pink = np.array([0.98823529, 0.05882353, 0.75294118])

gray = np.array([0.5, 0.5, 0.5])
dark_red = np.array([0.5, 0, 0])
dark_green = np.array([0, 0.5, 0])
dark_blue = np.array([0, 0, 0.5])

tan = np.array([0.82352941, 0.70588235, 0.54901961])
chartreuse = np.array([0.49803922, 1., 0.])
coral = np.array([1., 0.49803922, 0.31372549])


# a track buffer used only with picking tracks
track_buffer = []
# indices buffer for the tracks
ind_buffer = []
# tempory renderer used only with picking tracks
tmp_ren = None

if have_vtk:

    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
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


def ren():
    '''Create a renderer.

    Returns
    -------
    v : vtkRenderer() object
        Renderer.

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> import numpy as np
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10,3)]
    >>> c=fvtk.line(lines,fvtk.red)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)
    '''
    return vtk.vtkRenderer()


def add(ren, a):
    ''' Add a specific actor
    '''
    if isinstance(a, vtk.vtkVolume):
        ren.AddVolume(a)
    else:
        ren.AddActor(a)


def rm(ren, a):
    ''' Remove a specific actor
    '''
    ren.RemoveActor(a)


def clear(ren):
    ''' Remove all actors from the renderer
    '''
    ren.RemoveAllViewProps()


def rm_all(ren):
    ''' Remove all actors from the renderer
    '''
    clear(ren)


def _arrow(pos=(0, 0, 0), color=(1, 0, 0), scale=(1, 1, 1), opacity=1):
    ''' Internal function for generating arrow actors.
    '''
    arrow = vtk.vtkArrowSource()
    # arrow.SetTipLength(length)

    arrowm = vtk.vtkPolyDataMapper()

    if major_version <= 5:
        arrowm.SetInput(arrow.GetOutput())
    else:
        arrowm.SetInputData(arrow.GetOutput())

    arrowa = vtk.vtkActor()
    arrowa.SetMapper(arrowm)

    arrowa.GetProperty().SetColor(color)
    arrowa.GetProperty().SetOpacity(opacity)
    arrowa.SetScale(scale)

    return arrowa


def axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0), colorz=(0, 0, 1),
         opacity=1):
    ''' Create an actor with the coordinate system axes where  red = x, green = y, blue =z.
    '''

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


def _lookup(colors):
    ''' Internal function
    Creates a lookup table with given colors.

    Parameters
    ------------
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
              0=<b<=1,

    Returns
    ----------
    vtkLookupTable

    '''

    colors = np.asarray(colors, dtype=np.float32)

    if colors.ndim > 2:
        raise ValueError('Incorrect shape of array in colors')

    if colors.ndim == 1:
        N = 1

    if colors.ndim == 2:

        N = colors.shape[0]

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(N)
    lut.Build()

    if colors.ndim == 2:
        scalar = 0
        for (r, g, b) in colors:

            lut.SetTableValue(scalar, r, g, b, 1.0)
            scalar += 1
    if colors.ndim == 1:

        lut.SetTableValue(0, colors[0], colors[1], colors[2], 1.0)

    return lut


def pretty_line(lines, colors, linewidth=0.15, tube_sides=8):
    """ Uses streamtubes to visualize curves
    """
    points = vtk.vtkPoints()

    # Create the polyline.
    streamlines = vtk.vtkCellArray()

    cols = vtk.vtkUnsignedCharArray()
    cols.SetName("Cols")
    cols.SetNumberOfComponents(3)

    len_lines = len(lines)
    prior_line_shape = 0
    for i in range(len_lines):
        line = lines[i]
        streamlines.InsertNextCell(line.shape[0])
        for j in range(line.shape[0]):
            points.InsertNextPoint(*line[j])
            streamlines.InsertCellPoint(j + prior_line_shape)
            color = (255*colors[i]).astype('ubyte')
            cols.InsertNextTuple3(*color)
        prior_line_shape += line.shape[0]

    profileData = vtk.vtkPolyData()
    profileData.SetPoints(points)
    profileData.SetLines(streamlines)
    profileData.GetPointData().AddArray(cols)

    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(tube_sides)
    profileTubes.SetInput(profileData)
    profileTubes.SetRadius(linewidth)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInputConnection(profileTubes.GetOutputPort())
    profileMapper.ScalarVisibilityOn();
    profileMapper.SetScalarModeToUsePointFieldData()
    profileMapper.SelectColorArray("Cols")
    profileMapper.GlobalImmediateModeRenderingOn()

    profile = vtk.vtkLODActor()
    profile.SetMapper(profileMapper)
    #profile.GetProperty().SetDiffuseColor(banana)
    profile.GetProperty().SetSpecular(.3)
    profile.GetProperty().SetSpecularPower(10)
    profile.GetProperty().BackfaceCullingOn()
    profile.SetNumberOfCloudPoints(10**5)
    profile.GetProperty().SetPointSize(5)
    print(profile.GetNumberOfCloudPoints())


    return profile

def line(lines, colors, opacity=1, linewidth=1):
    ''' Create an actor for one or more lines.

    Parameters
    ------------
    lines :  list of arrays representing lines as 3d points  for example
            lines=[np.random.rand(10,3),np.random.rand(20,3)]
            represents 2 lines the first with 10 points and the second with 20 points in x,y,z coordinates.
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

    Returns
    ----------
    v : vtkActor object
        Line.

    Examples
    ----------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]
    >>> colors=np.random.rand(2,3)
    >>> c=fvtk.line(lines,colors)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)
    '''
    if not isinstance(lines, types.ListType):
        lines = [lines]

    points = vtk.vtkPoints()
    lines_ = vtk.vtkCellArray()
    linescalars = vtk.vtkFloatArray()

    # lookuptable=vtk.vtkLookupTable()
    lookuptable = _lookup(colors)

    scalarmin = 0
    if colors.ndim == 2:
        scalarmax = colors.shape[0] - 1
    if colors.ndim == 1:
        scalarmax = 0

    curPointID = 0

    m = (0.0, 0.0, 0.0)
    n = (1.0, 0.0, 0.0)

    scalar = 0
    # many colors
    if colors.ndim == 2:
        for Line in lines:

            inw = True
            mit = iter(Line)
            nit = iter(Line)
            next(nit)

            while(inw):

                try:
                    m = next(mit)
                    n = next(nit)

                    # scalar=sp.rand(1)

                    linescalars.SetNumberOfComponents(1)
                    points.InsertNextPoint(m)
                    linescalars.InsertNextTuple1(scalar)

                    points.InsertNextPoint(n)
                    linescalars.InsertNextTuple1(scalar)

                    lines_.InsertNextCell(2)
                    lines_.InsertCellPoint(curPointID)
                    lines_.InsertCellPoint(curPointID + 1)

                    curPointID += 2
                except StopIteration:
                    break

            scalar += 1
    # one color only
    if colors.ndim == 1:
        for Line in lines:

            inw = True
            mit = iter(Line)
            nit = iter(Line)
            next(nit)

            while(inw):

                try:
                    m = next(mit)
                    n = next(nit)

                    # scalar=sp.rand(1)

                    linescalars.SetNumberOfComponents(1)
                    points.InsertNextPoint(m)
                    linescalars.InsertNextTuple1(scalar)

                    points.InsertNextPoint(n)
                    linescalars.InsertNextTuple1(scalar)

                    lines_.InsertNextCell(2)
                    lines_.InsertCellPoint(curPointID)
                    lines_.InsertCellPoint(curPointID + 1)

                    curPointID += 2
                except StopIteration:
                    break

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines_)
    polydata.GetPointData().SetScalars(linescalars)

    mapper = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        mapper.SetInput(polydata)
    else:
        mapper.SetInputData(polydata)

    mapper.SetLookupTable(lookuptable)

    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(scalarmin, scalarmax)
    mapper.SetScalarModeToUsePointData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def dots(points, color=(1, 0, 0), opacity=1):
    '''
    Create one or more 3d dots(points) returns one actor handling all the points
    '''

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
    aPolyVertexGrid.InsertNextCell(aPolyVertex.GetCellType(), aPolyVertex.GetPointIds())

    aPolyVertexGrid.SetPoints(polyVertexPoints)
    aPolyVertexMapper = vtk.vtkDataSetMapper()
    if major_version <=  5:
        aPolyVertexMapper.SetInput(aPolyVertexGrid)
    else:
        aPolyVertexMapper.SetInputData(aPolyVertexGrid)
    aPolyVertexActor = vtk.vtkActor()
    aPolyVertexActor.SetMapper(aPolyVertexMapper)

    aPolyVertexActor.GetProperty().SetColor(color)
    aPolyVertexActor.GetProperty().SetOpacity(opacity)
    return aPolyVertexActor


def point(points, colors, opacity=1, point_radius=0.1, theta=3, phi=3):

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
            round(255 * colors[cnt_colors][0]), round(255 * colors[cnt_colors][1]), round(255 * colors[cnt_colors][2]))
        # scalars.InsertNextTuple3(255,255,255)
        cnt_colors += 1

    '''
    src = vtk.vtkDiskSource()
    src.SetRadialResolution(1)
    src.SetCircumferentialResolution(10)
    src.SetInnerRadius(0.0)
    src.SetOuterRadius(0.001)
    '''
    # src = vtk.vtkPointSource()
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

    mapper = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        mapper.SetInput(glyph.GetOutput())
    else:
        mapper.SetInputData(glyph.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def sphere(position=(0, 0, 0), radius=0.5, thetares=8, phires=8,
           color=(0, 0, 1), opacity=1, tessel=0):
    ''' Create a sphere actor
    '''
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetLatLongTessellation(tessel)

    sphere.SetThetaResolution(thetares)
    sphere.SetPhiResolution(phires)

    spherem = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        spherem.SetInput(sphere.GetOutput())
    else:
        spherem.SetInputData(sphere.GetOutput())
    spherea = vtk.vtkActor()
    spherea.SetMapper(spherem)
    spherea.SetPosition(position)
    spherea.GetProperty().SetColor(color)
    spherea.GetProperty().SetOpacity(opacity)

    return spherea


def ellipsoid(R=np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
              position=(0, 0, 0), thetares=20, phires=20, color=(0, 0, 1),
              opacity=1, tessel=0):
    ''' Create a ellipsoid actor.

    Stretch a unit sphere to make it an ellipsoid under a 3x3 translation
    matrix R.

    R = sp.array([[2, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1] ])
    '''

    Mat = sp.identity(4)
    Mat[0:3, 0:3] = R

    '''
    Mat=sp.array([[2, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0,  1]  ])
    '''
    mat = vtk.vtkMatrix4x4()

    for i in sp.ndindex(4, 4):

        mat.SetElement(i[0], i[1], Mat[i])

    radius = 1
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetLatLongTessellation(tessel)

    sphere.SetThetaResolution(thetares)
    sphere.SetPhiResolution(phires)

    trans = vtk.vtkTransform()

    trans.Identity()
    # trans.Scale(0.3,0.9,0.2)
    trans.SetMatrix(mat)
    trans.Update()

    transf = vtk.vtkTransformPolyDataFilter()
    transf.SetTransform(trans)

    if major_version <= 5:
        transf.SetInput(sphere.GetOutput())
    else:
        transf.SetInputData(sphere.GetOutput())
    transf.Update()

    spherem = vtk.vtkPolyDataMapper()
    if major_version <= 5:
        spherem.SetInput(transf.GetOutput())
    else:
        spherem.SetInputData(transf.GetOutput())

    spherea = vtk.vtkActor()
    spherea.SetMapper(spherem)
    spherea.SetPosition(position)
    spherea.GetProperty().SetColor(color)
    spherea.GetProperty().SetOpacity(opacity)
    # spherea.GetProperty().SetRepresentationToWireframe()

    return spherea


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
    ----------
    l : vtkActor object
        Label.

    Examples
    ----------
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
        point ``(-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,-vol.shape[2]/2.0+0.5)``.
    info : int {0,1}
        If 1 it prints out some info about the volume, the method and the
        dataset.
    trilinear : int {0,1}
        Use trilinear interpolation, default 1, gives smoother rendering. If
        you want faster interpolation use 0 (Nearest).
    maptype : int {0,1}
        The maptype is a very important parameter which affects the raycasting algorithm in use for the rendering.
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

    if colormap == None:

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
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0], vol.shape[1], vol.shape[2])
    # im.SetOrigin(0,0,0)
    # im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()

    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):

                im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

    if affine != None:

        aff = vtk.vtkMatrix4x4()
        aff.DeepCopy((affine[0, 0], affine[0, 1], affine[0, 2], affine[0, 3], affine[1, 0], affine[1, 1], affine[1, 2], affine[1, 3], affine[2, 0], affine[
                     2, 1], affine[2, 2], affine[2, 3], affine[3, 0], affine[3, 1], affine[3, 2], affine[3, 3]))
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
            changeFilter.SetOutputOrigin(-vol.shape[0] / 2.0 + 0.5, -vol.shape[1] / 2.0 + 0.5, -vol.shape[2] / 2.0 + 0.5)
            print('ChangeFilter ', changeFilter.GetOutputOrigin())

    opacity = vtk.vtkPiecewiseFunction()
    for i in range(opacitymap.shape[0]):
        opacity.AddPoint(opacitymap[i, 0], opacitymap[i, 1])

    color = vtk.vtkColorTransferFunction()
    for i in range(colormap.shape[0]):
        color.AddRGBPoint(colormap[i, 0], colormap[i, 1], colormap[i, 2], colormap[i, 3])

    if(maptype == 0):

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
        if affine == None:
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

        if affine == None:
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
    ''' Take a volume and draw surface contours for any any number of
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
    -----------
    ass : assembly of actors
        Representing the contour surfaces.

    Examples
    -------------
    >>> import numpy as np
    >>> from dipy.viz import fvtk
    >>> A=np.zeros((10,10,10))
    >>> A[3:-3,3:-3,3:-3]=1
    >>> r=fvtk.ren()
    >>> fvtk.add(r,fvtk.contour(A,levels=[1]))
    >>> #fvtk.show(r)

    '''

    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0], vol.shape[1], vol.shape[2])
    # im.SetOrigin(0,0,0)
    # im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()

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
        # ass=ass+[skin]

    return ass


def _cm2colors(colormap='Blues'):
    '''
    Colormaps from matplotlib
    ['Spectral', 'summer', 'RdBu', 'gist_earth', 'Set1', 'Set2', 'Set3', 'Dark2',
    'hot', 'PuOr_r', 'PuBuGn_r', 'RdPu', 'gist_ncar_r', 'gist_yarg_r', 'Dark2_r',
    'YlGnBu', 'RdYlBu', 'hot_r', 'gist_rainbow_r', 'gist_stern', 'cool_r', 'cool',
    'gray', 'copper_r', 'Greens_r', 'GnBu', 'gist_ncar', 'spring_r', 'gist_rainbow',
    'RdYlBu_r', 'gist_heat_r', 'OrRd_r', 'bone', 'gist_stern_r', 'RdYlGn', 'Pastel2_r',
    'spring', 'Accent', 'YlOrRd_r', 'Set2_r', 'PuBu', 'RdGy_r', 'spectral', 'flag_r', 'jet_r',
    'RdPu_r', 'gist_yarg', 'BuGn', 'Paired_r', 'hsv_r', 'YlOrRd', 'Greens', 'PRGn',
    'gist_heat', 'spectral_r', 'Paired', 'hsv', 'Oranges_r', 'prism_r', 'Pastel2', 'Pastel1_r',
     'Pastel1', 'gray_r', 'PuRd_r', 'Spectral_r', 'BuGn_r', 'YlGnBu_r', 'copper',
    'gist_earth_r', 'Set3_r', 'OrRd', 'PuBu_r', 'winter_r', 'jet', 'bone_r', 'BuPu',
    'Oranges', 'RdYlGn_r', 'PiYG', 'YlGn', 'binary_r', 'gist_gray_r', 'BuPu_r',
    'gist_gray', 'flag', 'RdBu_r', 'BrBG', 'Reds', 'summer_r', 'GnBu_r', 'BrBG_r',
    'Reds_r', 'RdGy', 'PuRd', 'Accent_r', 'Blues', 'Greys', 'autumn', 'PRGn_r', 'Greys_r',
    'pink', 'binary', 'winter', 'pink_r', 'prism', 'YlOrBr', 'Purples_r', 'PiYG_r', 'YlGn_r',
    'Blues_r', 'YlOrBr_r', 'Purples', 'autumn_r', 'Set1_r', 'PuOr', 'PuBuGn']

    '''
    try:
        from pylab import cm
    except ImportError:
        ImportError('pylab is not installed')

    blue = cm.datad[colormap]['blue']
    blue1 = [b[0] for b in blue]
    blue2 = [b[1] for b in blue]

    red = cm.datad[colormap]['red']
    red1 = [b[0] for b in red]
    red2 = [b[1] for b in red]

    green = cm.datad[colormap]['green']
    green1 = [b[0] for b in green]
    green2 = [b[1] for b in green]

    return red1, red2, green1, green2, blue1, blue2


def create_colormap(v, name='jet', auto=True):
    ''' Create colors from a specific colormap and return it
    as an array of shape (N,3) where every row gives the corresponding
    r,g,b value. The colormaps we use are similar with those of pylab.

    Parameters
    ----------
    v : (N,) array
        vector of values to be mapped in RGB colors according to colormap
    name : str. 'jet', 'blues', 'blue_red', 'accent'
        name of the colourmap
    auto : bool,
        if auto is True then v is interpolated to [0, 10] from v.min()
        to v.max()

    Notes
    -----
    If you want to add more colormaps here is what you could do. Go to
    this website http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
    see which colormap you need and then get in pylab using the cm.datad
    dictionary.

    e.g.::

          cm.datad['jet']

          {'blue': ((0.0, 0.5, 0.5),
                    (0.11, 1, 1),
                    (0.34000000000000002, 1, 1),
                    (0.65000000000000002, 0, 0),
                    (1, 0, 0)),
           'green': ((0.0, 0, 0),
                    (0.125, 0, 0),
                    (0.375, 1, 1),
                    (0.64000000000000001, 1, 1),
                    (0.91000000000000003, 0, 0),
                    (1, 0, 0)),
           'red': ((0.0, 0, 0),
                   (0.34999999999999998, 0, 0),
                   (0.66000000000000003, 1, 1),
                   (0.89000000000000001, 1, 1),
                   (1, 0.5, 0.5))}

    '''
    if v.ndim > 1:
        ValueError('This function works only with 1d arrays. Use ravel()')

    if auto:
        v = np.interp(v, [v.min(), v.max()], [0, 1])
    else:
        v = np.interp(v, [0, 1], [0, 1])

    if name == 'jet':
        # print 'jet'

        red = np.interp(v, [0, 0.35, 0.66, 0.89, 1], [0, 0, 1, 1, 0.5])
        green = np.interp(v, [0, 0.125, 0.375, 0.64, 0.91, 1], [0, 0, 1, 1, 0, 0])
        blue = np.interp(v, [0, 0.11, 0.34, 0.65, 1], [0.5, 1, 1, 0, 0])

    if name == 'blues':
        # cm.datad['Blues']
        # print 'blues'

        red = np.interp(
            v, [
                0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], [0.96862745285, 0.870588243008, 0.776470601559, 0.61960786581,
                                                                         0.419607847929, 0.258823543787, 0.129411771894, 0.0313725508749, 0.0313725508749])
        green = np.interp(
            v, [
                0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], [0.984313726425, 0.921568632126, 0.858823537827, 0.792156875134,
                                                                         0.68235296011, 0.572549045086, 0.443137258291, 0.317647069693, 0.188235297799])
        blue = np.interp(
            v, [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], [1.0, 0.96862745285, 0.937254905701, 0.882352948189,
                                                                         0.839215695858, 0.776470601559, 0.709803938866, 0.611764729023, 0.419607847929])

    if name == 'blue_red':
        # print 'blue_red'
        # red=np.interp(v,[],[])

        red = np.interp(v, [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], [0.0, 0.125, 0.25, 0.375, 0.5,
                        0.625, 0.75, 0.875, 1.0])
        green = np.zeros(red.shape)
        blue = np.interp(v, [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], [1.0, 0.875, 0.75, 0.625, 0.5,
                         0.375, 0.25, 0.125, 0.0])

        blue = green

    if name == 'accent':
        # print 'accent'
        red = np.interp(
            v, [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714,
                0.7142857142857143, 0.8571428571428571, 1.0],
            [0.49803921580314636, 0.7450980544090271, 0.99215686321258545, 1.0, 0.21960784494876862, 0.94117647409439087, 0.74901962280273438, 0.40000000596046448])
        green = np.interp(
            v, [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714,
                0.7142857142857143, 0.8571428571428571, 1.0],
            [0.78823530673980713, 0.68235296010971069, 0.75294119119644165, 1.0, 0.42352941632270813, 0.0078431377187371254, 0.35686275362968445, 0.40000000596046448])
        blue = np.interp(
            v, [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714,
                0.7142857142857143, 0.8571428571428571, 1.0],
            [0.49803921580314636, 0.83137255907058716, 0.52549022436141968, 0.60000002384185791, 0.69019609689712524, 0.49803921580314636, 0.090196080505847931, 0.40000000596046448])

    return np.vstack((red, green, blue)).T


def sphere_funcs(sphere_values, sphere, image=None, colormap='jet',
                     scale=2.2, norm=True, radial_scale=True):
    """Plot many morphed spheres simultaneously.

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
    if sphere_values.ndim == 1:
        sphere_values = sphere_values[None, None, None, :]
    if sphere_values.ndim == 2:
        sphere_values = sphere_values[None, None, :]
    if sphere_values.ndim == 3:
        sphere_values = sphere_values[None, :]
    if sphere_values.ndim > 4:
        raise ValueError("Wrong shape")

    grid_shape = np.array(sphere_values.shape[:3])
    faces = np.asarray(sphere.faces, dtype=int)
    vertices = sphere.vertices

    if sphere_values.shape[-1] != sphere.vertices.shape[0]:
        msg = 'Sphere.vertice.shape[0] should be the same as the'
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


def tensor(evals, evecs, scalar_colors=None, sphere=None, scale=2.2, norm=True):
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
    if evals.ndim == 1:
        evals = evals[None, None, None, :]
        evecs = evecs[None, None, None, :, :]
    if evals.ndim == 2:
        evals = evals[None, None, :]
        evecs = evecs[None, None, :, :]
    if evals.ndim == 3:
        evals = evals[None, :]
        evecs = evecs[None, :, :]
    if evals.ndim > 4:
        raise ValueError("Wrong shape")

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
        cfa = scalar_colors

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


def tube(point1=(0, 0, 0), point2=(1, 0, 0), color=(1, 0, 0), opacity=1, radius=0.1, capson=1, specular=1, sides=8):

    ''' Deprecated

    Wrap a tube around a line connecting point1 with point2 with a specific
    radius.

    '''
    points = vtk.vtkPoints()
    points.InsertPoint(0, point1[0], point1[1], point1[2])
    points.InsertPoint(1, point2[0], point2[1], point2[2])

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(2)

    lines.InsertCellPoint(0)
    lines.InsertCellPoint(1)

    profileData = vtk.vtkPolyData()
    profileData.SetPoints(points)
    profileData.SetLines(lines)

    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(sides)
    if major_version <= 5:
        profileTubes.SetInput(profileData)
    else:
        profileTubes.SetInputData(profileData)
    profileTubes.SetRadius(radius)

    if capson:
        profileTubes.SetCapping(1)
    else:
        profileTubes.SetCapping(0)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInputConnection(profileTubes.GetOutputPort())

    profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)
    profile.GetProperty().SetDiffuseColor(color)
    profile.GetProperty().SetSpecular(specular)
    profile.GetProperty().SetSpecularPower(30)
    profile.GetProperty().SetOpacity(opacity)

    return profile


def _closest_track(p, tracks):
    ''' Return the index of the closest track from tracks to point p
    '''

    d = []
    # enumt= enumerate(tracks)

    for (ind, t) in enumerate(tracks):
        for i in range(len(t[:-1])):

            d.append(
                (ind, np.sqrt(np.sum(np.cross((p - t[i]), (p - t[i + 1])) ** 2)) / np.sqrt(np.sum((t[i + 1] - t[i]) ** 2))))

    d = np.array(d)

    imin = d[:, 1].argmin()

    return int(d[imin, 0])


def crossing(a, ind, sph, scale, orient=False):
    """ visualize a volume of crossings

    Examples
    ----------
    See 'dipy/doc/examples/visualize_crossings.py' at :ref:`examples`

    """

    T = []
    Tor = []
    if a.ndim == 4 or a.ndim == 3:
        x, y, z = ind.shape[:3]
        for pos in np.ndindex(x, y, z):
            i, j, k = pos
            pos_ = np.array(pos)
            ind_ = ind[i, j, k]
            a_ = a[i, j, k]

            try:
                len(ind_)
            except TypeError:
                ind_ = [ind_]
                a_ = [a_]

            for (i, _i) in enumerate(ind_):
                T.append(pos_ + scale * a_[i] * np.vstack((sph[_i], -sph[_i])))
                if orient:
                    Tor.append(sph[_i])

    if a.ndim == 1:

        for (i, _i) in enumerate(ind):
                T.append(scale * a[i] * np.vstack((sph[_i], -sph[_i])))
                if orient:
                    Tor.append(sph[_i])
    if orient:
        return T, Tor
    return T


def slicer(ren, vol, voxsz=(1.0, 1.0, 1.0), affine=None, contours=1,
           planes=1, levels=[20, 30, 40], opacities=[0.8, 0.7, 0.3],
           colors=None, planesx=[20, 30], planesy=[30, 40], planesz=[20, 30]):
    ''' Slicer and contour rendering of 3d volumes

    Parameters
    ----------------
    vol : array, shape (N, M, K), dtype uint8
        An array representing the volumetric dataset that we want to visualize
        using volumetric rendering.
    voxsz : sequence of 3 floats
        Voxel size.
    affine : array, shape (4,4), default None
        As given by ``volumeimages``.
    contours : bool 1 to show contours
        Whether to show contours.
    planes : boolean 1 show planes
        Whether to show planes.
    levels : sequence
        Contour levels.
    opacities : sequence
        Opacity for every contour level.
    colors : None or sequence of 3-tuples
        Color for each contour level.
    planesx : (2,) array_like
        Saggital.
    planesy : (2,) array_like
        Coronal.
    planesz :
        Axial (2,) array_like

    Examples
    --------------
    >>> import numpy as np
    >>> from dipy.viz import fvtk
    >>> x, y, z = np.ogrid[-10:10:80j, -10:10:80j, -10:10:80j]
    >>> s = np.sin(x*y*z)/(x*y*z)
    >>> r=fvtk.ren()
    >>> #fvtk.slicer(r,s) #does showing too
    '''
    vol = np.interp(vol, xp=[vol.min(), vol.max()], fp=[0, 255])
    vol = vol.astype('uint8')

    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0], vol.shape[1], vol.shape[2])
    # im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    im.AllocateScalars()

    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):

                im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

    Contours = []
    for le in levels:
        # An isosurface, or contour value of 500 is known to correspond to the
        # skin of the patient. Once generated, a vtkPolyDataNormals filter is
        # is used to create normals for smooth surface shading during rendering.
        # The triangle stripper is used to create triangle strips from the
        # isosurface these render much faster on may systems.
        skinExtractor = vtk.vtkContourFilter()
        # skinExtractor.SetInputConnection(im.GetOutputPort())
        if major_version <= 5:
            skinExtractor.SetInput(im)
        else:
            skinExtractor.SetInputData(im)
        skinExtractor.SetValue(0, le)
        skinNormals = vtk.vtkPolyDataNormals()
        skinNormals.SetInputConnection(skinExtractor.GetOutputPort())
        skinNormals.SetFeatureAngle(60.0)
        skinStripper = vtk.vtkStripper()
        skinStripper.SetInputConnection(skinNormals.GetOutputPort())
        skinMapper = vtk.vtkPolyDataMapper()
        skinMapper.SetInputConnection(skinStripper.GetOutputPort())
        skinMapper.ScalarVisibilityOff()
        skin = vtk.vtkActor()
        skin.SetMapper(skinMapper)
        if colors == None:
            skin.GetProperty().SetDiffuseColor(1, .49, .25)
        else:
            colorskin = colors[le]
            skin.GetProperty().SetDiffuseColor(colorskin[0], colorskin[1], colorskin[2])
        skin.GetProperty().SetSpecular(.3)
        skin.GetProperty().SetSpecularPower(20)

        Contours.append(skin)

    # An outline provides context around the data.
    outlineData = vtk.vtkOutlineFilter()
    # outlineData.SetInputConnection(im.GetOutputPort())
    if major_version <= 5:
        outlineData.SetInput(im)
    else:
        outlineData.SetInputData(im)
    mapOutline = vtk.vtkPolyDataMapper()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(mapOutline)
    outline.GetProperty().SetColor(1, 0, 0)

    # Now we are creating three orthogonal planes passing through the
    # volume. Each plane uses a different texture map and therefore has
    # diferent coloration.

    # Start by creatin a black/white lookup table.
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(vol.min(), vol.max())
    lut.SetSaturationRange(0, 0)
    lut.SetHueRange(0, 0)
    lut.SetValueRange(0, 1)
    lut.SetRampToLinear()
    lut.Build()

    x1, x2, y1, y2, z1, z2 = im.GetExtent()

    # print x1,x2,y1,y2,z1,z2

    # Create the first of the three planes. The filter vtkImageMapToColors
    # maps the data through the corresponding lookup table created above.
    # The vtkImageActor is a type of vtkProp and conveniently displays an
    # image on a single quadrilateral plane. It does this using texture
    # mapping and as a result is quite fast. (Note: the input image has to
    # be unsigned char values, which the vtkImageMapToColors produces.)
    # Note also that by specifying the DisplayExtent, the pipeline
    # requests data of this extent and the vtkImageMapToColors only
    # processes a slice of data.
    planeColors = vtk.vtkImageMapToColors()
    # saggitalColors.SetInputConnection(im.GetOutputPort())
    if major_version <= 5:
        planeColors.SetInput(im)
    else:
        planeColors.SetInputData(im)
    planeColors.SetLookupTable(lut)
    planeColors.Update()

    saggitals = []
    for x in planesx:

        saggital = vtk.vtkImageActor()
        if major_version <= 5:
            saggital.SetInput(planeColors.GetOutput())
        else:
            saggital.SetInputData(planeColors.GetOutput())
        saggital.SetDisplayExtent(x, x, y1, y2, z1, z2)

        saggitals.append(saggital)

    axials = []
    for z in planesz:
        axial = vtk.vtkImageActor()
        if major_version <= 5:
            axial.SetInput(planeColors.GetOutput())
        else:
            axial.SetInputData(planeColors.GetOutput())
        axial.SetDisplayExtent(x1, x2, y1, y2, z, z)
        axials.append(axial)

    coronals = []
    for y in planesy:
        coronal = vtk.vtkImageActor()
        if major_version <= 5:
            coronal.SetInput(planeColors.GetOutput())
        else:
            coronal.SetInputData(planeColors.GetOutput())
        coronal.SetDisplayExtent(x1, x2, y, y, z1, z2)
        coronals.append(coronal)

    # It is convenient to create an initial view of the data. The FocalPoint
    # and Position form a vector direction. Later on (ResetCamera() method)
    # this vector is used to position the camera to look at the data in
    # this direction.
    aCamera = vtk.vtkCamera()
    aCamera.SetViewUp(0, 0, -1)
    aCamera.SetPosition(0, 1, 0)
    aCamera.SetFocalPoint(0, 0, 0)
    aCamera.ComputeViewPlaneNormal()

    # saggital.SetOpacity(0.1)

    # Actors are added to the renderer.
    ren.AddActor(outline)
    if planes:
        for sag in saggitals:
            ren.AddActor(sag)
        for ax in axials:
            ren.AddActor(ax)
        for cor in coronals:
            ren.AddActor(cor)

    if contours:
        cnt = 0
        for actor in Contours:
            actor.GetProperty().SetOpacity(opacities[cnt])
            ren.AddActor(actor)
            cnt += 1

    # Turn off bone for this example.
    # bone.VisibilityOff()

    # Set skin to semi-transparent.

    # An initial camera view is created.  The Dolly() method moves
    # the camera towards the FocalPoint, thereby enlarging the image.
    ren.SetActiveCamera(aCamera)
    ren.ResetCamera()
    aCamera.Dolly(1.5)

    # Set a background color for the renderer and set the size of the
    # render window (expressed in pixels).
    ren.SetBackground(0, 0, 0)
    # renWin.SetSize(640, 480)

    # Note that when camera movement occurs (as it does in the Dolly()
    # method), the clipping planes often need adjusting. Clipping planes
    # consist of two planes: near and far along the view direction. The
    # near plane clips out objects in front of the plane the far plane
    # clips out objects behind the plane. This way only what is drawn
    # between the planes is actually rendered.
    # ren.ResetCameraClippingRange()

    # return ren

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ren.ResetCameraClippingRange()

    # Interact with the data.
    iren.Initialize()
    renWin.Render()
    iren.Start()


def annotatePick(object, event):
    ''' Create a Python function to create the text for the
    text mapper used to display the results of picking.
    '''
    global picker, textActor, textMapper, track_buffer

    if picker.GetCellId() < 0:
        textActor.VisibilityOff()
    else:
        if len(track_buffer) != 0:

            selPt = picker.GetSelectionPoint()
            pickPos = picker.GetPickPosition()

            closest = _closest_track(np.array([pickPos[0], pickPos[1], pickPos[2]]), track_buffer)

            if major_version <= 5:
                textMapper.SetInput("(%.6f, %.6f, %.6f)" % pickPos)
            else:
                textMapper.SetInputData("(%.6f, %.6f, %.6f)" % pickPos)
            textActor.SetPosition(selPt[:2])
            textActor.VisibilityOn()

            label(tmp_ren, text=str(ind_buffer[closest]), pos=(track_buffer[closest][0][0], track_buffer[
                  closest][0][1], track_buffer[closest][0][2]))

            tmp_ren.AddActor(line(track_buffer[closest], golden, opacity=1))


def show(ren, title='Dipy', size=(300, 300), png_magnify=1):
    ''' Show window

    Notes
    -----
    To save a screenshot press's' and check your current directory
    for ``fvtk.png``.

    Parameters
    ------------
    ren : vtkRenderer() object
        As returned from function ``ren()``.
    title : string
        A string for the window title bar.
    size : (int, int)
        ``(width, height)`` of the window
    png_magnify : int
        Number of times to magnify the screenshot.

    Notes
    -----
    If you want to:

    * navigate in the the 3d world use the left - middle - right mouse buttons
    * reset the screen press 'r'
    * save a screenshot press 's'
    * quit press 'q'

    See also
    ---------
    dipy.viz.fvtk.record

    Examples
    ----------
    >>> import numpy as np
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]
    >>> colors=np.array([[0.2,0.2,0.2],[0.8,0.8,0.8]])
    >>> c=fvtk.line(lines,colors)
    >>> fvtk.add(r,c)
    >>> l=fvtk.label(r)
    >>> fvtk.add(r,l)
    >>> #fvtk.show(r)

    See also
    ----------
    dipy.viz.fvtk.record

    '''

    ren.ResetCamera()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    #window.SetAAFrames(6)
    window.SetWindowName(title)
    window.SetSize(size[0], size[1])
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)
    iren.SetPicker(picker)

    def key_press(obj, event):

        key = obj.GetKeySym()
        if key == 's' or key == 'S':
            print('Saving image...')
            renderLarge = vtk.vtkRenderLargeImage()
            if major_version <= 5:
                renderLarge.SetInput(ren)
            else:
                renderLarge.SetInputData(ren)
            renderLarge.SetMagnification(png_magnify)
            renderLarge.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetInputConnection(renderLarge.GetOutputPort())
            writer.SetFileName('fvtk.png')
            writer.Write()
            print('Look for fvtk.png in your current working directory.')

    iren.AddObserver('KeyPressEvent', key_press)
    iren.SetInteractorStyle(style)
    iren.Initialize()
    picker.Pick(85, 126, 0, ren)
    window.Render()
    iren.Start()

    # window.RemoveAllObservers()
    # ren.SetRenderWindow(None)
    window.RemoveRenderer(ren)
    ren.SetRenderWindow(None)


def record(ren=None, cam_pos=None, cam_focal=None, cam_view=None,
           out_path=None, path_numbering=False, n_frames=10, az_ang=10,
           magnification=1, size=(300, 300), bgr_color=(0, 0, 0),
           verbose=False):
    ''' This will record a video of your scene

    Records a video as a series of ``.png`` files of your scene by rotating the
    azimuth angle az_angle in every frame.

    Parameters
    -----------
    ren : vtkRenderer() object
        as returned from function ren()
    cam_pos : None or sequence (3,), optional
        camera position
    cam_focal : None or sequence (3,), optional
        camera focal point
    cam_view : None or sequence (3,), optional
        camera view up
    out_path : str, optional
        output directory for the frames
    path_numbering : bool
        when recording it changes out_path ot out_path + str(frame number)
    n_frames : int, optional
        number of frames to save, default 10
    az_ang : float, optional
        azimuthal angle of camera rotation.
    magnification : int, optional
        how much to magnify the saved frame

    Examples
    ---------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> a=fvtk.axes()
    >>> fvtk.add(r,a)
    >>> #uncomment below to record
    >>> #fvtk.record(r)
    >>> #check for new images in current directory
    '''
    if ren == None:
        ren = vtk.vtkRenderer()
    ren.SetBackground(bgr_color)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(size[0], size[1])
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # ren.GetActiveCamera().Azimuth(180)

    ren.ResetCamera()

    renderLarge = vtk.vtkRenderLargeImage()
    if major_version <= 5:
        renderLarge.SetInput(ren)
    else:
        renderLarge.SetInputData(ren)
    renderLarge.SetMagnification(magnification)
    renderLarge.Update()

    writer = vtk.vtkPNGWriter()
    ang = 0

    if cam_pos != None:
        cx, cy, cz = cam_pos
        ren.GetActiveCamera().SetPosition(cx, cy, cz)
    if cam_focal != None:
        fx, fy, fz = cam_focal
        ren.GetActiveCamera().SetFocalPoint(fx, fy, fz)
    if cam_view != None:
        ux, uy, uz = cam_view
        ren.GetActiveCamera().SetViewUp(ux, uy, uz)

    cam = ren.GetActiveCamera()
    if verbose:
        print('Camera Position (%.2f,%.2f,%.2f)' % cam.GetPosition())
        print('Camera Focal Point (%.2f,%.2f,%.2f)' % cam.GetFocalPoint())
        print('Camera View Up (%.2f,%.2f,%.2f)' % cam.GetViewUp())

    for i in range(n_frames):
        ren.GetActiveCamera().Azimuth(ang)
        renderLarge = vtk.vtkRenderLargeImage()
        if major_version <= 5:
            renderLarge.SetInput(ren)
        else:
            renderLarge.SetInputData(ren)
        renderLarge.SetMagnification(magnification)
        renderLarge.Update()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        # filename='/tmp/'+str(3000000+i)+'.png'
        if path_numbering:
            if out_path == None:
                filename = str(1000000 + i) + '.png'
            else:
                filename = out_path + str(1000000 + i) + '.png'
        else:
            filename = out_path
        writer.SetFileName(filename)
        writer.Write()

        ang = +az_ang


if __name__ == "__main__":
    pass
