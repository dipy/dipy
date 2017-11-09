import os
import numpy as np
from dipy.viz import actor, window
import numpy.testing as npt
from dipy.testing.decorators import xvfb_it

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False


@npt.dec.skipif(not actor.have_vtk or not actor.have_vtk_colors or skip_it)
@xvfb_it
def test_renderer():

    ren = window.Renderer()

    npt.assert_equal(ren.size(), (0, 0))

    # background color for renderer (1, 0.5, 0)
    # 0.001 added here to remove numerical errors when moving from float
    # to int values
    bg_float = (1, 0.501, 0)

    # that will come in the image in the 0-255 uint scale
    bg_color = tuple((np.round(255 * np.array(bg_float))).astype('uint8'))

    ren.background(bg_float)
    # window.show(ren)
    arr = window.snapshot(ren)

    report = window.analyze_snapshot(arr,
                                     bg_color=bg_color,
                                     colors=[bg_color, (0, 127, 0)])
    npt.assert_equal(report.objects, 0)
    npt.assert_equal(report.colors_found, [True, False])

    axes = actor.axes()
    ren.add(axes)
    # window.show(ren)

    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 1)

    ren.rm(axes)
    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 0)

    window.add(ren, axes)
    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 1)

    ren.rm_all()
    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 0)

    ren2 = window.renderer(bg_float)
    ren2.background((0, 0, 0.))

    report = window.analyze_renderer(ren2)
    npt.assert_equal(report.bg_color, (0, 0, 0))

    ren2.add(axes)

    report = window.analyze_renderer(ren2)
    npt.assert_equal(report.actors, 3)

    window.rm(ren2, axes)
    report = window.analyze_renderer(ren2)
    npt.assert_equal(report.actors, 0)


@npt.dec.skipif(not actor.have_vtk or not actor.have_vtk_colors or skip_it)
@xvfb_it
def test_active_camera():
    renderer = window.Renderer()
    renderer.add(actor.axes(scale=(1, 1, 1)))

    renderer.reset_camera()
    renderer.reset_clipping_range()

    direction = renderer.camera_direction()
    position, focal_point, view_up = renderer.get_camera()

    renderer.set_camera((0., 0., 1.), (0., 0., 0), view_up)

    position, focal_point, view_up = renderer.get_camera()
    npt.assert_almost_equal(np.dot(direction, position), -1)

    renderer.zoom(1.5)

    new_position, _, _ = renderer.get_camera()

    npt.assert_array_almost_equal(position, new_position)

    renderer.zoom(1)

    # rotate around focal point
    renderer.azimuth(90)

    position, _, _ = renderer.get_camera()

    npt.assert_almost_equal(position, (1.0, 0.0, 0))

    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, colors=[(255, 0, 0)])
    npt.assert_equal(report.colors_found, [True])

    # rotate around camera's center
    renderer.yaw(90)

    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, colors=[(0, 0, 0)])
    npt.assert_equal(report.colors_found, [True])

    renderer.yaw(-90)
    renderer.elevation(90)

    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(arr, colors=(0, 255, 0))
    npt.assert_equal(report.colors_found, [True])

    renderer.set_camera((0., 0., 1.), (0., 0., 0), view_up)

    # vertical rotation of the camera around the focal point
    renderer.pitch(10)
    renderer.pitch(-10)

    # rotate around the direction of projection
    renderer.roll(90)

    # inverted normalized distance from focal point along the direction
    # of the camera

    position, _, _ = renderer.get_camera()
    renderer.dolly(0.5)
    new_position, _, _ = renderer.get_camera()
    npt.assert_almost_equal(position[2], 0.5 * new_position[2])


@npt.dec.skipif(not actor.have_vtk or not actor.have_vtk_colors or skip_it)
@xvfb_it
def test_parallel_projection():

    ren = window.Renderer()
    axes = actor.axes()
    axes2 = actor.axes()
    axes2.SetPosition((2, 0, 0))

    # Add both axes.
    ren.add(axes, axes2)

    # Put the camera on a angle so that the
    # camera can show the difference between perspective
    # and parallel projection
    ren.set_camera((1.5, 1.5, 1.5))
    ren.GetActiveCamera().Zoom(2)

    # window.show(ren, reset_camera=True)
    ren.reset_camera()
    arr = window.snapshot(ren)

    ren.projection('parallel')
    # window.show(ren, reset_camera=False)
    arr2 = window.snapshot(ren)
    # Because of the parallel projection the two axes
    # will have the same size and therefore occupy more
    # pixels rather than in perspective projection were
    # the axes being further will be smaller.
    npt.assert_equal(np.sum(arr2 > 0) > np.sum(arr > 0), True)


@npt.dec.skipif(not actor.have_vtk or not actor.have_vtk_colors or skip_it)
@xvfb_it
def test_order_transparent():

    renderer = window.Renderer()

    lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
             np.array([[-1, 1, 0.], [1, 1, 0.]])]
    colors = np.array([[1., 0., 0.], [0., .5, 0.]])
    stream_actor = actor.streamtube(lines, colors, linewidth=0.3, opacity=0.5)

    renderer.add(stream_actor)

    renderer.reset_camera()

    # green in front
    renderer.elevation(90)
    renderer.camera().OrthogonalizeViewUp()
    renderer.reset_clipping_range()

    renderer.reset_camera()

    not_xvfb = os.environ.get("TEST_WITH_XVFB", False)

    if not_xvfb:
        arr = window.snapshot(renderer, fname='green_front.png',
                              offscreen=True, order_transparent=False)
    else:
        arr = window.snapshot(renderer, fname='green_front.png',
                              offscreen=False, order_transparent=False)

    # therefore the green component must have a higher value (in RGB terms)
    npt.assert_equal(arr[150, 150][1] > arr[150, 150][0], True)

    # red in front
    renderer.elevation(-180)
    renderer.camera().OrthogonalizeViewUp()
    renderer.reset_clipping_range()

    if not_xvfb:
        arr = window.snapshot(renderer, fname='red_front.png',
                              offscreen=True, order_transparent=True)
    else:
        arr = window.snapshot(renderer, fname='red_front.png',
                              offscreen=False, order_transparent=True)

    # therefore the red component must have a higher value (in RGB terms)
    npt.assert_equal(arr[150, 150][0] > arr[150, 150][1], True)


if __name__ == '__main__':

    npt.run_module_suite()
