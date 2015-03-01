from __future__ import division, print_function, absolute_import

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

#import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

if have_vtk:

    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()


def renderer(background=None):
    """ Create a renderer.

    Parameters
    ----------
    background : tuple
        Initial background color of renderer

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
    >>> c=fvtk.line(lines, fvtk.colors.red)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)
    """
    ren = vtk.vtkRenderer()
    if background is not None:
        ren.SetBackground(background)

    return ren


ren = renderer()

def add(ren, a):
    """ Add a specific actor
    """
    if isinstance(a, vtk.vtkVolume):
        ren.AddVolume(a)
    else:
        ren.AddActor(a)


def rm(ren, a):
    """ Remove a specific actor
    """
    ren.RemoveActor(a)


def clear(ren):
    """ Remove all actors from the renderer
    """
    ren.RemoveAllViewProps()


def rm_all(ren):
    """ Remove all actors from the renderer
    """
    clear(ren)


def show(ren, title='Dipy', size=(300, 300), png_magnify=1):
    """ Show window

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

    """

    ren.ResetCamera()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    # window.SetAAFrames(6)
    window.SetWindowName(title)
    window.SetSize(size[0], size[1])
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)
    #iren.SetPicker(picker)

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
    #picker.Pick(85, 126, 0, ren)
    window.Render()
    iren.Start()

    # window.RemoveAllObservers()
    # ren.SetRenderWindow(None)
    window.RemoveRenderer(ren)
    ren.SetRenderWindow(None)


def record(ren=None, cam_pos=None, cam_focal=None, cam_view=None,
           out_path=None, path_numbering=False, n_frames=1, az_ang=10,
           magnification=1, size=(300, 300), verbose=False):
    """ This will record a video of your scene

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
        number of frames to save, default 1
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
    """
    if ren is None:
        ren = vtk.vtkRenderer()

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

    if cam_pos is not None:
        cx, cy, cz = cam_pos
        ren.GetActiveCamera().SetPosition(cx, cy, cz)
    if cam_focal is not None:
        fx, fy, fz = cam_focal
        ren.GetActiveCamera().SetFocalPoint(fx, fy, fz)
    if cam_view is not None:
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
        renderLarge.SetInput(ren)
        renderLarge.SetMagnification(magnification)
        renderLarge.Update()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        # filename='/tmp/'+str(3000000+i)+'.png'
        if path_numbering:
            if out_path is None:
                filename = str(1000000 + i) + '.png'
            else:
                filename = out_path + str(1000000 + i) + '.png'
        else:
            filename = out_path
        writer.SetFileName(filename)
        writer.Write()

        ang = +az_ang
