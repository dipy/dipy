from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import ndimage
from copy import copy

try:
    import Tkinter as tkinter
except ImportError:
    import tkinter

try:
    import tkFileDialog as filedialog
except ImportError:
    from tkinter import filedialog

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

from dipy import __version__ as dipy_version
from dipy.utils.six import string_types
from dipy.viz.actor import Container


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
    vtkRenderer = vtk.vtkRenderer
else:
    vtkRenderer = object

if have_imread:
    from scipy.misc import imread


class Renderer(vtkRenderer):
    """ Your scene class

    This is an important object that is responsible for preparing objects
    e.g. actors and volumes for rendering. This is a more pythonic version
    of ``vtkRenderer`` proving simple methods for adding and removing actors
    but also it provides access to all the functionality
    available in ``vtkRenderer`` if necessary.
    """

    def background(self, color):
        """ Set a background color
        """
        self.SetBackground(color)

    def add(self, *actors):
        """ Add an actor to the renderer
        """
        for actor in actors:
            if isinstance(actor, Container):
                actor.add_to_renderer(self)
            elif isinstance(actor, vtk.vtkVolume):
                self.AddVolume(actor)
            elif isinstance(actor, vtk.vtkActor2D):
                self.AddActor2D(actor)
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
        """ Reset the camera to an automatic position given by the engine.
        """
        self.ResetCamera()

    def reset_camera_tight(self, margin_factor=1.02):
        """ Resets camera so the content fit tightly within the window.

        Parameters
        ----------
        margin_factor : float (optional)
            Margin added around the content. Default: 1.02.

        Notes
        -----
        This reset function works best with
        ``:func:dipy.interactor.InteractorStyleImageAndTrackballActor``.
        """
        self.ComputeAspect()
        cam = self.GetActiveCamera()
        aspect = self.GetAspect()

        X1, X2, Y1, Y2, Z1, Z2 = self.ComputeVisiblePropBounds()
        width, height = X2-X1, Y2-Y1
        center = np.array((X1 + width/2., Y1 + height/2., 0))

        angle = np.pi*cam.GetViewAngle()/180.
        dist = max(width/aspect[0], height) / np.sin(angle/2.) / 2.
        position = center + np.array((0, 0, dist*margin_factor))

        cam.SetViewUp(0, 1, 0)
        cam.SetPosition(*position)
        cam.SetFocalPoint(*center)
        self.ResetCameraClippingRange(X1, X2, Y1, Y2, Z1, Z2)

        parallelScale = max(width/aspect[0], height) / 2.
        cam.SetParallelScale(parallelScale*margin_factor)

    def reset_clipping_range(self):
        self.ResetCameraClippingRange()

    def camera(self):
        return self.GetActiveCamera()

    def get_camera(self):
        cam = self.GetActiveCamera()
        return cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()

    def camera_info(self):
        cam = self.camera()
        print('# Active Camera')
        print('   Position (%.2f, %.2f, %.2f)' % cam.GetPosition())
        print('   Focal Point (%.2f, %.2f, %.2f)' % cam.GetFocalPoint())
        print('   View Up (%.2f, %.2f, %.2f)' % cam.GetViewUp())

    def set_camera(self, position=None, focal_point=None, view_up=None):
        if position is not None:
            self.GetActiveCamera().SetPosition(*position)
        if focal_point is not None:
            self.GetActiveCamera().SetFocalPoint(*focal_point)
        if view_up is not None:
            self.GetActiveCamera().SetViewUp(*view_up)
        self.ResetCameraClippingRange()

    def size(self):
        """ Renderer size"""
        return self.GetSize()

    def zoom(self, value):
        """ In perspective mode, decrease the view angle by the specified
        factor. In parallel mode, decrease the parallel scale by the specified
        factor. A value greater than 1 is a zoom-in, a value less than 1 is a
        zoom-out.
        """
        self.GetActiveCamera().Zoom(value)

    def azimuth(self, angle):
        """ Rotate the camera about the view up vector centered at the focal
        point. Note that the view up vector is whatever was set via SetViewUp,
        and is not necessarily perpendicular to the direction of projection.
        The result is a horizontal rotation of the camera.
        """
        self.GetActiveCamera().Azimuth(angle)

    def yaw(self, angle):
        """ Rotate the focal point about the view up vector, using the camera's
        position as the center of rotation. Note that the view up vector is
        whatever was set via SetViewUp, and is not necessarily perpendicular
        to the direction of projection. The result is a horizontal rotation of
        the scene.
        """
        self.GetActiveCamera().Yaw(angle)

    def elevation(self, angle):
        """ Rotate the camera about the cross product of the negative of the
        direction of projection and the view up vector, using the focal point
        as the center of rotation. The result is a vertical rotation of the
        scene.
        """
        self.GetActiveCamera().Elevation(angle)

    def pitch(self, angle):
        """ Rotate the focal point about the cross product of the view up
        vector and the direction of projection, using the camera's position as
        the center of rotation. The result is a vertical rotation of the
        camera.
        """
        self.GetActiveCamera().Pitch(angle)

    def roll(self, angle):
        """ Rotate the camera about the direction of projection. This will
        spin the camera about its axis.
        """
        self.GetActiveCamera().Roll(angle)

    def dolly(self, value):
        """ Divide the camera's distance from the focal point by the given
        dolly value. Use a value greater than one to dolly-in toward the focal
        point, and use a value less than one to dolly-out away from the focal
        point.
        """
        self.GetActiveCamera().Dolly(value)

    def camera_direction(self):
        """ Get the vector in the direction from the camera position to the
        focal point. This is usually the opposite of the ViewPlaneNormal, the
        vector perpendicular to the screen, unless the view is oblique.
        """
        return np.array(self.GetActiveCamera().GetDirectionOfProjection())


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

    root = tkinter.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(filetypes=file_types)
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

    root = tkinter.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(initialfile=initial_file,
                                             defaultextension=default_ext,
                                             filetypes=file_types)
    return file_path


class ShowManager(object):
    """ This class is the interface between the renderer, the window and the
    interactor.
    """

    def __init__(self, ren, title='DIPY', size=(300, 300),
                 png_magnify=1, reset_camera=True, order_transparent=False,
                 interactor_style='trackball', picker_pos=(10, 10, 0),
                 picker_tol=0.002):

        """ Manages the visualization pipeline

        Parameters
        ----------
        ren : Renderer() or vtkRenderer()
            The scene that holds all the actors.
        title : string
            A string for the window title bar.
        size : (int, int)
            ``(width, height)`` of the window
        png_magnify : int
            Number of times to magnify the screenshot. This can be used to save
            high resolution screenshots when pressing 's' inside the window.
        reset_camera : bool
            Default is True. You can change this option to False if you want to
            keep the camera as set before calling this function.
        order_transparent : bool
            True is useful when you want to order transparent
            actors according to their relative position to the camera. The
            default option which is False will order the actors according to
            the order of their addition to the Renderer().
        interactor_style : str or vtkInteractorStyle
            If str then if 'trackball' then vtkInteractorStyleTrackballCamera()
            is used or if 'image' then vtkInteractorStyleImage() is used (no
            rotation). Otherwise you can input your own interactor style.
        picker_pos : tuple
        picker_tol : float

        Attributes
        ----------
        ren : vtkRenderer()
        iren : vtkRenderWindowInteractor()
        style : vtkInteractorStyle()
        window : vtkRenderWindow()

        Methods
        -------
        initialize()
        render()
        start()
        add_window_callback()

        Notes
        -----
        Default interaction keys for

        * 3d navigation are with left, middle and right mouse dragging
        * resetting the camera press 'r'
        * saving a screenshot press 's'
        * for quiting press 'q'

        Examples
        --------
        >>> from dipy.viz import actor, window
        >>> renderer = window.Renderer()
        >>> renderer.add(actor.axes())
        >>> showm = window.ShowManager(renderer)
        >>> # showm.initialize()
        >>> # showm.render()
        >>> # start()
        """

        self.ren = ren
        self.title = title
        self.size = size
        self.png_magnify = png_magnify
        self.reset_camera = reset_camera
        self.order_transparent = order_transparent
        self.interactor_style = interactor_style
        self.picker_pos = picker_pos
        self.picker_tol = picker_tol
        self.timers = []

        if self.reset_camera:
            self.ren.ResetCamera()

        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(ren)

        if self.title == 'DIPY':
            self.window.SetWindowName(title + ' ' + dipy_version)
        else:
            self.window.SetWindowName(title)
        self.window.SetSize(size[0], size[1])

        if self.order_transparent:

            # Use a render window with alpha bits
            # as default is 0 (false))
            self.window.SetAlphaBitPlanes(True)

            # Force to not pick a framebuffer with a multisample buffer
            # (default is 8)
            self.window.SetMultiSamples(0)

            # Choose to use depth peeling (if supported)
            # (default is 0 (false)):
            self.ren.UseDepthPeelingOn()

            # Set depth peeling parameters
            # Set the maximum number of rendering passes (default is 4)
            ren.SetMaximumNumberOfPeels(4)

            # Set the occlusion ratio (initial value is 0.0, exact image):
            ren.SetOcclusionRatio(0.0)

        if self.interactor_style == 'image':
            self.style = vtk.vtkInteractorStyleImage()
        elif self.interactor_style == 'trackball':
            self.style = vtk.vtkInteractorStyleTrackballCamera()
        else:
            self.style = interactor_style

        self.iren = vtk.vtkRenderWindowInteractor()
        self.style.SetCurrentRenderer(self.ren)
        self.style.SetInteractor(self.iren)  # Hack: this allows the Python version of this method to be called.
        self.iren.SetInteractorStyle(self.style)
        self.iren.SetRenderWindow(self.window)

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
                                            default_ext='.png',
                                            file_types=file_types)
                if filepath == '':
                    print('No file was provided in the dialog')
                else:
                    writer = vtk.vtkPNGWriter()
                    writer.SetInputConnection(renderLarge.GetOutputPort())
                    writer.SetFileName(filepath)
                    writer.Write()
                    print('File ' + filepath + ' is saved.')

        self.iren.AddObserver('KeyPressEvent', key_press_standard)

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(self.picker_tol)
        self.iren.SetPicker(self.picker)

    def initialize(self):
        """ Initialize interaction
        """
        self.iren.Initialize()

        i, j, k = self.picker_pos
        self.picker.Pick(i, j, k, self.ren)

    def render(self):
        """ Renders only once
        """
        self.window.Render()

    def start(self):
        """ Starts interaction
        """
        try:
            self.iren.Start()
        except AttributeError:
            self.__init__(self.ren, self.title, size=self.size,
                          png_magnify=self.png_magnify,
                          reset_camera=self.reset_camera,
                          order_transparent=self.order_transparent,
                          interactor_style=self.interactor_style)
            self.initialize()
            self.render()
            self.iren.Start()

        # window.RemoveAllObservers()
        # ren.SetRenderWindow(None)

        self.window.RemoveRenderer(self.ren)
        self.ren.SetRenderWindow(None)
        del self.iren
        del self.window

    def add_window_callback(self, win_callback):
        """ Add window callbacks
        """
        self.window.AddObserver(vtk.vtkCommand.ModifiedEvent, win_callback)

    def add_picker_callback(self, picker_callback):
        self.picker.AddObserver("EndPickEvent", picker_callback)

    def add_timer_callback(self, repeat, duration, timer_callback):
        self.iren.AddObserver("TimerEvent", timer_callback)

        if repeat:
            timer_id = self.iren.CreateRepeatingTimer(duration)
        else:
            timer_id = self.iren.CreateOneShotTimer(duration)
        self.timers.append(timer_id)


def show(ren, title='DIPY', size=(300, 300),
         png_magnify=1, reset_camera=True, order_transparent=False):
    """ Show window with current renderer

    Parameters
    ------------
    ren : Renderer() or vtkRenderer()
        The scene that holds all the actors.
    title : string
        A string for the window title bar.
    size : (int, int)
        ``(width, height)`` of the window
    png_magnify : int
        Number of times to magnify the screenshot. This can be used to save
        high resolution screenshots when pressing 's' inside the window.
    reset_camera : bool
        Default is True. You can change this option to False if you want to
        keep the camera as set before calling this function.
    order_transparent : bool
        True is useful when you want to order transparent
        actors according to their relative position to the camera. The default
        option which is False will order the actors according to the order of
        their addition to the Renderer().

    Notes
    -----
    Default interaction keys for

    * 3d navigation are with left, middle and right mouse dragging
    * resetting the camera press 'r'
    * saving a screenshot press 's'
    * for quiting press 'q'

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
    ---------
    dipy.viz.window.record
    dipy.viz.window.snapshot
    """

    show_manager = ShowManager(ren, title, size,
                               png_magnify, reset_camera, order_transparent)
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
    fname : str or None
        Save PNG file. If None return only an array without saving PNG.
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
        bg_color = im[0, 0]
        background = np.dot(bg_color, weights)

        if strel is None:
            strel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])

        labels, objects = ndimage.label(gray != background, strel)
        report.labels = labels
        report.objects = objects

    return report


class MovieWriter():

    def __init__(self, fname, window, encoder='ffmpeg',
                 bit_rate=None, bit_rate_tol=None, frame_rate=None,
                 compression=False, compression_quality=None):

        self.wif = vtk.vtkWindowToImageFilter()
        self.wif.SetInput(window)
        self.wif.ReadFrontBufferOff()
        self.wif.Update()

        self.writer_alive = True
        self.bit_rate = bit_rate
        self.bit_rate_tol = bit_rate_tol
        self.frame_rate = frame_rate
        self.compression = compression
        self.compression_quality = compression_quality

        if encoder == 'ffmpeg':
            self.writer = vtk.vtkFFMPEGWriter()
            self.writer.SetInputConnection(self.wif.GetOutputPort())
            self.writer.SetFileName(fname)
            if bit_rate is not None:
                self.writer.SetBitRate(bit_rate)
                self.writer.SetBitRateTolerance(bit_rate_tol)
            if frame_rate is not None:
                self.writer.SetRate(frame_rate)
            self.writer.SetCompression(compression)
            if compression_quality is not None:
                self.writer.SetQuality(compression_quality)

        if encoder == 'png':
            raise ValueError('PNG writing not currently supported')

        if encoder == 'gif':
            raise ValueError('GIF writing not currently supported')

    def start(self):
        if self.writer_alive:
            self.writer.Start()

    def write(self):
        self.wif.Modified()
        self.writer.Write()

    def end(self):
        self.writer.End()
        self.write_alive = False

    def __del__(self):
        if self.writer_alive:
            self.writer.End()
