'''The FVTK module implements simple visualization functions using VTK.

A window can have one or more renderers. Actors (drawing primitives, such as
pheres, lines, points, etc.) are added to the renderer, which then visualizes
them.

Examples
---------
>>> from dipy.viz import fvtk
>>> r = fvtk.renderer()
>>> a = fvtk.axes()
>>> r.add(a)
>>> # r.show()

For more information on VTK, see the examples at

http://www.vtk.org/Wiki/VTK/Tutorials/External_Tutorials

'''
from __future__ import division, print_function, absolute_import

from dipy.utils.six.moves import range

from ._vtk import setup_vtk
vtk, have_vtk, version, major_version = setup_vtk()

if have_vtk:
    # Create a cell picker.
    picker = vtk.vtkCellPicker()

from ..utils.optpkg import optional_package
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')

from .fvtk_actors import *

class renderer(vtk.vtkRenderer):
    def __init__(self):
        '''Create a renderer object.

        Returns
        -------
        v : vtkRenderer() object
            Renderer.

        Examples
        --------
        >>> from dipy.viz import fvtk
        >>> import numpy as np
        >>> r = fvtk.renderer()
        >>> lines = [np.random.rand(10, 3)]
        >>> c = fvtk.line(lines, fvtk.colors.red)
        >>> r.add(c)
        >>> # r.show()
        '''
        pass

    def add(self, a):
        """Add an actor to the renderer.

        Parameters
        ----------
        a : vtk.VtkActor
            These actors are produced by the `axes`, `streamtube`, `line`,
            `dots`, `point`, `label`, `volume`, `contour`, `sphere_funcs`,
            `tensor`, `slicer`
        """
        if isinstance(a, vtk.vtkVolume):
            self.AddVolume(a)
        else:
            self.AddActor(a)

    def rm(self, a):
        """Remove a specific actor from the renderer.

        Parameters
        ----------
        a : vtk.VtkActor
            Actor to remove.
        """
        self.RemoveActor(a)

    def clear(self):
        """Remove all actors from the renderer.

        """
        self.RemoveAllViewProps()

    def show(self, title='Dipy', size=(300, 300), png_magnify=1):
        """Show window.

        Notes
        -----
        To save a screenshot press 's'.  The screenshots are stored as
        ``fvtk0000000.png``, ``fvtk0000001.png``, etc.

        Parameters
        ------------
        title : string
            A string for the window title bar.
        size : (width, height) tuple of int
            Size of the window in pixels.
        png_magnify : int
            Number of times to magnify the screenshot.

        Notes
        -----
        If you want to:

        * navigate in the the 3d world use the left - middle - right
          mouse buttons
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
        >>> r = fvtk.renderer()
        >>> lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
        >>> colors = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
        >>> r.add(fvtk.line(lines, colors))
        >>> r.add(fvtk.label('Origin'))
        >>> #fvtk.show(r)

        See also
        ----------
        dipy.viz.fvtk.record

        """
        self.ResetCamera()
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
        self.SetRenderWindow(None)


# --- Backwards compatible API
ren = renderer


def add(ren, a):
    ren.add(a)


def rm(ren, a):
    ren.rm(a)


def clear(ren):
    ren.clear()


def rm_all(ren):
    ren.clear()


def show(ren):
    ren.show()

# ------------


def camera(ren, pos=None, focal=None, viewup=None, verbose=True):
    """Change the active camera.

    Parameters
    ----------
    ren : vtkRenderer
        Renderer.
    pos : tuple
        ``(x, y, z)`` position of the camera.
    focal : tuple
        ``(x, y, z)`` focal point.
    viewup : tuple
        ``(x, y, z)`` viewup vector.
    verbose : bool
        Print information about the camera.

    Returns
    -------
    c : vtkCamera
        Active camera.
    """

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
            print('Camera New Position (%.2f, %.2f, %.2f)' %
                  cam.GetPosition())
            print('Camera New Focal Point (%.2f, %.2f, %.2f)' %
                  cam.GetFocalPoint())
            print('Camera New View Up (%.2f, %.2f, %.2f)' %
                  cam.GetViewUp())

    return cam


def record(ren=None, out_path=None, cam_pos=None, cam_focal=None,
           cam_view=None, path_numbering=False, n_frames=1, az_ang=10,
           magnification=1, size=(300, 300), verbose=False):
    """Record one or more snapshots of the current scene.

    Multiple views of the scene are saved as ``.png`` snapshots.  After each
    snapshot, the scene is rotated by a given azimuth angle.

    Parameters
    -----------
    ren : vtkRenderer() object
        The renderer, as returned by ``renderer()``.
    out_path : str, optional
        Output directory and filename for the frames, e.g. 'output/snapshot'.
        If `path_numbering` is enabled, output filenames will be
        of the form `output/snapshot0000005.png'.
    cam_pos : tuple (x, y, z), optional
        Camera position (see notes below).
    cam_focal : tuple (x, y, z), optional
        Camera focal point (see notes below).
    cam_view : tuple (x, y, z), optional
        Camera upward orientation vector (see notes below).
    path_numbering : bool
        If True, append the current frame number to the output filename.
    n_frames : int, optional
        Number of frames to save (default 1).
    az_ang : float, optional
        Azimuthal angle of camera rotation in degrees, applied after each
        snapshot.
    magnification : int, optional
        Magnifiation applied to each saved frame.

    Notes
    -----
    The camera view is determined by three vectors:

    - `cam_pos`: The location of the camera in three-dimensional space.
    - `cam_focal`: The point at which the camera is aimed (XXX check)
    - `cam_view`: A vector pointing along the vertical axis of the camera.
                  I.e., if the camera is horizontally aligned with the X axis,
                  pointing in the direction of the Y axis, then `cam_view` is
                  along the Z axis.

    Examples
    ---------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.renderer()
    >>> a=fvtk.axes()
    >>> fvtk.add(r,a)
    >>> #uncomment below to record
    >>> #fvtk.record(r)
    >>> #check for new images in current directory
    """
    if ren is None:
        ren = renderer()

    if out_path is None:
        out_path = 'snapshot'
    elif out_path.endswith('.png'):
        out_path = out_path[:-4]

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

        if path_numbering:
            filename = out_path + '%07d' % i
        else:
            filename = out_path
        filename += '.png'

        writer.SetFileName(filename)
        writer.Write()

        ang = +az_ang
