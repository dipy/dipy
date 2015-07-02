from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import ndimage
import Tkinter
import tkFileDialog

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

from dipy import __version__ as dipy_version
from dipy.utils.six import string_types


# import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')
_, have_imread, _ = optional_package('Image')

if have_vtk:
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
    from vtk.util.numpy_support import vtk_to_numpy

if have_imread:
    from scipy.misc import imread


class Renderer(vtk.vtkRenderer):
    """ The key rendering preparation object

    This is an important object that is responsible for preparing objects
    e.g. actors and volumes for rendering. This is a more pythonic version
    of ``vtkRenderer`` proving simple methods for adding and removing actors
    but also it provides access to all the functionality
    available in ``vtkRenderer``.

    """

    def background(self, color):
        """ Set a background color
        """
        self.SetBackground(color)

    def add(self, actor):
        """ Add an actor to the renderer
        """
        if isinstance(actor, vtk.vtkVolume):
            self.AddVolume(actor)
        else:
            self.AddActor(actor)

    def rm(self, actor):
        """ Remove a specific actor
        """
        self.RemoveActor(actor)

    def clear(self):
        """ Remove all actors from the renderer
        """
        self.RemoveAllViewProps()

    def rm_all(self):
        """ Remove all actors from the renderer
        """
        self.RemoveAllViewProps()

    def projection(self, proj_type='perspective'):
        """ Deside between parallel or perspective projection

        Parameters
        ----------
        proj_type : str
            Can be 'parallel' or 'perspective' (default).

        """
        if proj_type == 'parallel':
            self.GetActiveCamera().ParallelProjectionOn()
        else:
            self.GetActiveCamera().ParallelProjectionOff()

    def reset_camera(self):
        """ Allow the renderer to reset the camera
        """
        self.ResetCamera()


def renderer(background=None):
    """ Create a renderer.

    Parameters
    ----------
    background : tuple
        Initial background color of renderer

    Returns
    -------
    v : Renderer

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
    ren = Renderer()
    if background is not None:
        ren.SetBackground(background)

    return ren

if have_vtk:
    ren = renderer


def add(ren, a):
    """ Add a specific actor
    """
    ren.add(a)


def rm(ren, a):
    """ Remove a specific actor
    """
    ren.rm(a)


def clear(ren):
    """ Remove all actors from the renderer
    """
    ren.clear()


def rm_all(ren):
    """ Remove all actors from the renderer
    """
    ren.rm_all()


def open_file_dialog(file_types=[("All files", "*")]):
    """ Simple Tk file dialog for opening files

    Parameters
    ----------
    file_types : tuples of tuples
        Accepted file types.

    Returns
    -------
    file_paths : sequence of str
        Returns the full paths of all selected files
    """

    root = Tkinter.Tk()
    root.withdraw()
    file_paths = tkFileDialog.askopenfilenames(filetypes=file_types)
    return file_paths


def save_file_dialog(initial_file='dipy.png', default_ext='.png',
                     file_types=(("PNG file", "*.png"), ("All Files", "*.*"))):
    """ Simple Tk file dialog for saving a file

    Parameters
    ----------
    initial_file : str
        For example ``dipy.png``.
    default_ext : str
        Default extension to appear in the save dialog.
    file_types : tuples of tuples
        Accepted file types.

    Returns
    -------
    filepath : str
        Complete filename of saved file
    """

    root = Tkinter.Tk()
    root.withdraw()
    file_path = tkFileDialog.asksaveasfilename(initialfile=initial_file,
                                               defaultextension=default_ext,
                                               filetypes=file_types)
    return file_path


class ShowManager(object):

    def __init__(self, ren, title='Dipy', size=(300, 300),
                 png_magnify=1, reset_camera=True):

        self.title = title
        self.size = size
        self.png_magnify = png_magnify

        if reset_camera:
            ren.ResetCamera()

        window = vtk.vtkRenderWindow()
        window.AddRenderer(ren)
        # window.SetAAFrames(6)
        if title == 'Dipy':
            window.SetWindowName(title + ' ' + dipy_version)
        else:
            window.SetWindowName(title)
        window.SetSize(size[0], size[1])

        style = vtk.vtkInteractorStyleTrackballCamera()
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(window)

        def key_press_standard(obj, event):

            key = obj.GetKeySym()
            if key == 's' or key == 'S':
                print('Saving image...')
                renderLarge = vtk.vtkRenderLargeImage()
                if major_version <= 5:
                    renderLarge.SetInput(ren)
                else:
                    renderLarge.SetInput(ren)
                renderLarge.SetMagnification(png_magnify)
                renderLarge.Update()

                file_types = (("PNG file", "*.png"), ("All Files", "*.*"))
                filepath = save_file_dialog(initial_file='dipy.png',
                                            default_extension='.png',
                                            filetypes=file_types)
                if filepath == '':
                    print('No file was provided in the dialog')
                else:
                    writer = vtk.vtkPNGWriter()
                    writer.SetInputConnection(renderLarge.GetOutputPort())
                    writer.SetFileName(filepath)
                    writer.Write()
                    print('File ' + filepath + ' is saved.')

        self.window = window
        self.ren = ren
        self.iren = iren
        self.style = style

        self.iren.AddObserver('KeyPressEvent', key_press_standard)

        self.iren.SetInteractorStyle(self.style)

    def initialize(self):
        self.iren.Initialize()
        # picker.Pick(85, 126, 0, ren)

    def render(self):
        self.window.Render()

    def start(self):
        self.iren.Start()
        # window.RemoveAllObservers()
        # ren.SetRenderWindow(None)
        self.window.RemoveRenderer(self.ren)
        self.ren.SetRenderWindow(None)
        del self.iren
        del self.window

    def add_window_callback(self, win_callback):
        self.window.AddObserver(vtk.vtkCommand.ModifiedEvent, win_callback)
        self.window.Render()


def show(ren, title='Dipy', size=(300, 300),
         png_magnify=1, reset_camera=True):
    """ Show window with current renderer


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
    dipy.viz.window.record
    dipy.viz.window.snapshot

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

    show_manager = ShowManager(ren, title, size,
                               png_magnify, reset_camera)
    show_manager.initialize()
    show_manager.render()
    show_manager.start()


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
    size : (int, int)
        ``(width, height)`` of the window
    verbose : bool
        print information about the camera
    sleep_time : float
        Creates a small delay in seconds so that the renderer has enough
        time to save the figure correctly.

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

    if ren is None:
        ren.ResetCamera()

    renderLarge = vtk.vtkRenderLargeImage()
    if major_version <= 5:
        renderLarge.SetInput(ren)
    else:
        renderLarge.SetInput(ren)
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
        print('Camera Position (%.2f, %.2f, %.2f)' % cam.GetPosition())
        print('Camera Focal Point (%.2f, %.2f, %.2f)' % cam.GetFocalPoint())
        print('Camera View Up (%.2f, %.2f, %.2f)' % cam.GetViewUp())

    for i in range(n_frames):
        ren.GetActiveCamera().Azimuth(ang)
        renderLarge = vtk.vtkRenderLargeImage()
        renderLarge.SetInput(ren)
        renderLarge.SetMagnification(magnification)
        renderLarge.Update()
        writer.SetInputConnection(renderLarge.GetOutputPort())

        if path_numbering:
            if out_path is None:
                filename = str(i).zfill(6) + '.png'
            else:
                filename = out_path + str(i).zfill(6) + '.png'
        else:
            filename = out_path
        writer.SetFileName(filename)
        writer.Write()

        ang = +az_ang


def snapshot(ren, fname=None, size=(300, 300)):
    """ Saves a snapshot of the renderer in a file or in memory

    Parameters
    -----------
    ren : vtkRenderer
        as returned from function renderer()
    fname : str
        Save PNG file.
    size : (int, int)
        ``(width, height)`` of the window

    Returns
    -------
    arr : array
        Color array of size (width, height, 3) where the last dimension
        holds the RGB values.
    """

    width, height = size

    graphics_factory = vtk.vtkGraphicsFactory()
    graphics_factory.SetOffScreenOnlyMode(1)
    graphics_factory.SetUseMesaClasses(1)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(ren)
    render_window.SetSize(width, height)
    render_window.Render()

    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    h, w, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = vtk_to_numpy(vtk_array).reshape(h, w, components)

    if fname is None:
        return arr

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()
    return arr


def analyze_renderer(ren):

    class ReportRenderer(object):
        bg_color = None

    report = ReportRenderer()

    report.bg_color = ren.GetBackground()
    report.collection = ren.GetActors()
    report.actors = report.collection.GetNumberOfItems()

    report.collection.InitTraversal()
    report.actors_classnames = []
    for i in range(report.actors):
        class_name = report.collection.GetNextActor().GetClassName()
        report.actors_classnames.append(class_name)

    return report


def analyze_snapshot(im, bg_color=(0, 0, 0), colors=None,
                     find_objects=True,
                     strel=None):
    """ Analyze snapshot from memory or file

    Parameters
    ----------
    im: str or array
        If string then the image is read from a file otherwise the image is
        read from a numpy array. The array is expected to be of shape (X, Y, 3)
        where the last dimensions are the RGB values.
    colors: tuple (3,) or list of tuples (3,)
        List of colors to search in the image
    find_objects: bool
        If True it will calculate the number of objects that are different
        from the background and return their position in a new image.
    strel: 2d array
        Structure element to use for finding the objects.

    Returns
    -------
    report : ReportSnapshot
        This is an object with attibutes like ``colors_found`` that give
        information about what was found in the current snapshot array ``im``.

    """
    if isinstance(im, string_types):
        im = imread(im)

    class ReportSnapshot(object):
        objects = None
        labels = None
        colors_found = False

    report = ReportSnapshot()

    if colors is not None:
        if isinstance(colors, tuple):
            colors = [colors]
        flags = [False] * len(colors)
        for (i, col) in enumerate(colors):
            # find if the current color exist in the array
            flags[i] = np.any(np.all(im == col, axis=-1))

        report.colors_found = flags

    if find_objects is True:
        weights = [0.299, 0.587, 0.144]
        gray = np.dot(im[..., :3], weights)
        background = np.dot(bg_color, weights)

        if strel is None:
            strel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])

        labels, objects = ndimage.label(gray != background, strel)
        report.labels = labels
        report.objects = objects

    return report
