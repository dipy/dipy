import os
import numpy as np

from dipy.viz import actor, window

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory
from dipy.tracking.streamline import center_streamlines, transform_streamlines
from dipy.align.tests.test_streamlinear import fornix_streamlines
from dipy.data import get_sphere


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_slicer():

    renderer = window.renderer()

    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.slicer(data, affine)
    window.add(renderer, slicer)
    # window.show(renderer)

    # copy pixels in numpy array directly
    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)

    # The slicer can cut directly a smaller part of the image
    slicer.SetDisplayExtent(10, 30, 10, 30, 35, 35)
    slicer.Update()
    renderer.ResetCamera()

    window.add(renderer, slicer)

    # save pixels in png file not a numpy array
    with TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'slice.png')
        # window.show(renderer)
        arr = window.snapshot(renderer, fname)
        report = window.analyze_snapshot(fname, find_objects=True)
        npt.assert_equal(report.objects, 1)

    npt.assert_raises(ValueError, actor.slicer, np.ones(10))

    renderer.clear()

    rgb = np.zeros((30, 30, 30, 3))
    rgb[..., 0] = 1.
    rgb_actor = actor.slicer(rgb)

    renderer.add(rgb_actor)

    renderer.reset_camera()
    renderer.reset_clipping_range()

    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, colors=[(255, 0, 0)])
    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report.colors_found, [True])

    lut = actor.colormap_lookup_table(scale_range=(0, 255),
                                      hue_range=(0.4, 1.),
                                      saturation_range=(1, 1.),
                                      value_range=(0., 1.))
    renderer.clear()
    slicer_lut = actor.slicer(data, lookup_colormap=lut)

    slicer_lut.display(10, None, None)
    slicer_lut.display(None, 10, None)
    slicer_lut.display(None, None, 10)

    slicer_lut2 = slicer_lut.copy()
    slicer_lut2.display(None, None, 10)
    renderer.add(slicer_lut2)

    renderer.reset_clipping_range()

    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_streamtube_and_line_actors():

    renderer = window.renderer()

    line1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    line2 = line1 + np.array([0.5, 0., 0.])

    lines = [line1, line2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    c = actor.line(lines, colors, linewidth=3)
    window.add(renderer, c)

    c = actor.line(lines, colors, spline_subdiv=5, linewidth=3)
    window.add(renderer, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    arr = window.snapshot(renderer)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])

    # as before with splines
    c2 = actor.streamtube(lines, colors, spline_subdiv=5, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    arr = window.snapshot(renderer)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)

    npt.assert_equal(report.objects, 4)
    npt.assert_equal(report.colors_found, [True, True])


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_bundle_maps():

    renderer = window.renderer()
    bundle = fornix_streamlines()
    bundle, shift = center_streamlines(bundle)

    mat = np.array([[1, 0, 0, 100],
                    [0, 1, 0, 100],
                    [0, 0, 1, 100],
                    [0, 0, 0, 1.]])

    bundle = transform_streamlines(bundle, mat)

    # metric = np.random.rand(*(200, 200, 200))
    metric = 100 * np.ones((200, 200, 200))

    # add lower values
    metric[100, :, :] = 100 * 0.5

    # create a nice orange-red colormap
    lut = actor.colormap_lookup_table(scale_range=(0., 100.),
                                      hue_range=(0., 0.1),
                                      saturation_range=(1, 1),
                                      value_range=(1., 1))

    line = actor.line(bundle, metric, linewidth=0.1, lookup_colormap=lut)
    window.add(renderer, line)
    window.add(renderer, actor.scalar_bar(lut, ' '))

    report = window.analyze_renderer(renderer)

    npt.assert_almost_equal(report.actors, 1)
    # window.show(renderer)

    renderer.clear()

    nb_points = np.sum([len(b) for b in bundle])
    values = 100 * np.random.rand(nb_points)
    # values[:nb_points/2] = 0

    line = actor.streamtube(bundle, values, linewidth=0.1, lookup_colormap=lut)
    renderer.add(line)
    # window.show(renderer)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')

    renderer.clear()

    colors = np.random.rand(nb_points, 3)
    # values[:nb_points/2] = 0

    line = actor.line(bundle, colors, linewidth=2)
    renderer.add(line)
    # window.show(renderer)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')
    # window.show(renderer)

    arr = window.snapshot(renderer)
    report2 = window.analyze_snapshot(arr)
    npt.assert_equal(report2.objects, 1)

    # try other input options for colors
    renderer.clear()
    actor.line(bundle, (1., 0.5, 0))
    actor.line(bundle, np.arange(len(bundle)))
    actor.line(bundle)
    colors = [np.random.rand(*b.shape) for b in bundle]
    actor.line(bundle, colors=colors)


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_odf_slicer():

    sphere = get_sphere('symmetric362')

    # use memory maps
    # odfs = np.ones((10, 10, 10, sphere.vertices.shape[0]))

    shape = (11, 11, 11, sphere.vertices.shape[0])

    odfs = np.memmap('test.mmap', dtype='float64', mode='w+',
                     shape=shape)

    odfs[:] = 1
    # odfs = np.random.rand(10, 10, 10, sphere.vertices.shape[0])

    affine = np.eye(4)
    renderer = window.renderer()

    odf_actor = actor.odf_slicer(odfs, affine,
                                 mask=None, sphere=sphere, scale=.25,
                                 colormap='jet')

    fa = 0. * np.random.rand(*odfs.shape[:3])
    fa[:, 0, :] = 1.
    fa[:, -1, :] = 1.
    fa[0, :, :] = 1.
    fa[-1, :, :] = 1.
    fa[5, 5, 5] = 1

    fa_actor = actor.slicer(fa, affine)
    fa_actor.display(None, None, 5)

    renderer.add(fa_actor)
    renderer.add(odf_actor)
    renderer.reset_camera()
    renderer.reset_clipping_range()

    for k in range(5, 6):
        I, J, K = odfs.shape[:3]

        odf_actor.display_extent(0, I, 0, J, k, k + 1)
        odf_actor.GetProperty().SetOpacity(0.2)
        window.show(renderer, reset_camera=False)

def fig2data (fig):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels
    and return it.

    Look also here
    http://wiki.scipy.org/Cookbook/Matplotlib/VTK_Integration

    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis = 2)
    return buf


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_figure():

    #A = np.zeros((200, 100), dtype=np.uchar)
    #A[100, 50] = 255

    fname = '/home/eleftherios/.dipy/icons/icomoon/camera.png'
    from dipy.viz import actor, window

    #figure_actor = actor.figure(fname)
    #1/0
    renderer = window.renderer()
#    figure_actor.SetPosition(50, 50, 0)
#
#    renderer.add(figure_actor)
#
#    window.show(renderer)
#
#    renderer.clear()
#

    # create RGBA array
    A = 255 * np.ones((100, 200, 4), dtype=np.ubyte)
    A[:, 100] = np.array([255, 0, 0, 255])
    print(A.dtype)
    #A = np.swapaxes(A, 0, 2)
    figure_actor = actor.figure(A, interpolation='nearest')
    #figure_actor = actor.slicer(A)

    renderer.add(figure_actor)
    window.show(renderer)

    def fig2data(fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )

        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
        buf.shape = ( w, h,4 )

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = numpy.roll ( buf, 3, axis = 2 )
        return buf




if __name__ == "__main__":

    # npt.run_module_suite()
    # test_odf_slicer()
    test_figure()
