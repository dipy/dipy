import os
import numpy as np

from dipy.viz import actor, window, utils

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

    mask = np.ones(odfs.shape[:3])
    mask[:4, :4, :4] = 0

    odf_actor = actor.odf_slicer(odfs, affine,
                                 mask=mask, sphere=sphere, scale=.25,
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

    for k in range(0, 5):
        I, J, K = odfs.shape[:3]

        odf_actor.display_extent(0, I, 0, J, k, k + 1)
        odf_actor.GetProperty().SetOpacity(0.6)
        # window.show(renderer, reset_camera=False)


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
@npt.dec.skipif(not utils.have_mpl)
def test_figure():

    renderer = window.renderer()

    # create RGBA rectangle of width height 100 and width  200
    # with one red line in the middle
    A = 255 * np.ones((100, 200, 4), dtype=np.ubyte)
    A[:, 100] = np.array([255, 0, 0, 255])
    figure_actor = actor.figure(A, interpolation='nearest')
    renderer.add(figure_actor)

    renderer.reset_camera()
    renderer.zoom(3.5)

    # window.show(renderer, reset_camera=False)
    snap_arr = window.snapshot(renderer)

    npt.assert_array_equal(snap_arr[150, 150], [255, 0, 0])
    npt.assert_array_equal(snap_arr[160, 150], [255, 0, 0])
    npt.assert_array_equal(snap_arr[150, 160], [255, 255, 255])

    renderer.clear()

    # create a nice fill plot with matplotlib and show it in the VTK scene
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1)
    y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

    fig = plt.figure(figsize=(1000/300, 800/300), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_title('Fill plot')
    ax.fill(x, y, 'r')
    ax.grid(True)

    arr = utils.matplotlib_figure_to_numpy(fig, dpi=300, transparent=True)
    plt.close(fig)

    renderer.clear()
    renderer.background((0.7, 0.7, 0.7))

    figure_actor = actor.figure(arr, interpolation='cubic')

    renderer.add(figure_actor)

    axes_actor = actor.axes((50, 50, 50))
    axes_actor.SetPosition(500, 400, -50)

    # renderer.add(axes_actor)
    # window.show(renderer)
    renderer.reset_camera()

    props = renderer.GetViewProps()
    print(props.GetNumberOfItems())
    props.InitTraversal()
    npt.assert_equal(props.GetNextProp().GetClassName(), 'vtkImageActor')

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.bg_color, (0.7, 0.7, 0.7))


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_text_3d():
    ren = window.Renderer()

    grid_line = actor.line([np.array([[-1, -1, 0],
                                      [1, -1, 0],
                                      [1, 1, 0],
                                      [-1, 1, 0],
                                      [-1, -1, 0]], dtype=float),
                            np.array([[0, -1, 0.],
                                      [0, 1, 0]], dtype=float),
                            np.array([[-1, 0, 0.],
                                      [1, 0, 0]], dtype=float)], colors=(1., 1., 1.))
    ren.add(grid_line)

    ren.add(actor.text_3d("TestA", position=(-1, 1, 0), color=(1, 0, 0), font_size=0.2,
                          justification="left", vertical_justification="top"))
    ren.add(actor.text_3d("TestB", position=(0, 1, 0), color=(0, 1, 0), font_size=0.2,
                          justification="center", vertical_justification="top"))
    ren.add(actor.text_3d("TestC", position=(1, 1, 0), color=(0, 0, 1), font_size=0.2,
                          justification="right", vertical_justification="top"))

    ren.add(actor.text_3d("TestA", position=(-1, 0, 0), color=(1, 1, 0), font_size=0.2,
                          justification="left", vertical_justification="middle"))
    ren.add(actor.text_3d("TestB", position=(0, 0, 0), color=(0, 1, 1), font_size=0.2,
                          justification="center", vertical_justification="middle"))
    ren.add(actor.text_3d("TestC", position=(1, 0, 0), color=(1, 0, 1), font_size=0.2,
                          justification="right", vertical_justification="middle"))

    ren.add(actor.text_3d("TestA", position=(-1, -1, 0), color=(1, 0, 1), font_size=0.2,
                          justification="left", vertical_justification="bottom"))
    ren.add(actor.text_3d("TestB", position=(0, -1, 0), color=(1, 1, 0), font_size=0.2,
                          justification="center", vertical_justification="bottom"))
    ren.add(actor.text_3d("TestC", position=(1, -1, 0), color=(0, 1, 1), font_size=0.2,
                          justification="right", vertical_justification="bottom"))

    ren.add(actor.axes())
    ren.reset_camera()

    show_m = window.ShowManager(ren)
    show_m.start()

    # Multi-lines
    ren = window.Renderer()

    grid_line = actor.line([np.array([[-1, -1, 0],
                                      [1, -1, 0],
                                      [1, 1, 0],
                                      [-1, 1, 0],
                                      [-1, -1, 0]], dtype=float),
                            np.array([[0, -1, 0.],
                                      [0, 1, 0]], dtype=float),
                            np.array([[-1, 0, 0.],
                                      [1, 0, 0]], dtype=float)], colors=(1., 1., 1.))
    ren.add(grid_line)

    ren.add(actor.text_3d("TestA\nlines\nvery long long!", position=(-1, 1, 0), color=(1, 0, 0), font_size=0.2,
                          justification="left", vertical_justification="top"))
    ren.add(actor.text_3d("TestB", position=(0, 1, 0), color=(0, 1, 0), font_size=0.2,
                          justification="center", vertical_justification="top"))
    ren.add(actor.text_3d("TestC", position=(1, 1, 0), color=(0, 0, 1), font_size=0.2,
                          justification="right", vertical_justification="top"))

    ren.add(actor.text_3d("TestA", position=(-1, 0, 0), color=(1, 1, 0), font_size=0.2,
                          justification="left", vertical_justification="middle"))
    ren.add(actor.text_3d("TestB\nlines\nvery long long!", position=(0, 0, 0), color=(0, 1, 1), font_size=0.2,
                          justification="center", vertical_justification="middle"))
    ren.add(actor.text_3d("TestC", position=(1, 0, 0), color=(1, 0, 1), font_size=0.2,
                          justification="right", vertical_justification="middle"))

    ren.add(actor.text_3d("TestA", position=(-1, -1, 0), color=(1, 0, 1), font_size=0.2,
                          justification="left", vertical_justification="bottom"))
    ren.add(actor.text_3d("TestB", position=(0, -1, 0), color=(1, 1, 0), font_size=0.2,
                          justification="center", vertical_justification="bottom"))
    ren.add(actor.text_3d("TestC\nlines\nvery long long!", position=(1, -1, 0), color=(0, 1, 1), font_size=0.2,
                          justification="right", vertical_justification="bottom"))

    ren.add(actor.axes())
    ren.reset_camera()

    show_m = window.ShowManager(ren)
    show_m.start()

if __name__ == "__main__":
    test_text_3d()
    #npt.run_module_suite()
