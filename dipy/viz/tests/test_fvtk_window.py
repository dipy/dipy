import numpy as np
from dipy.viz import actor
from dipy.viz import window
from dipy.viz import widget
from dipy.data import fetch_viz_icons, read_viz_icons
import numpy.testing as npt


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_window_module():

    ren = window.Renderer()

    ren.background((1, 0.5, 0))

    window.show(ren)

    arr = window.snapshot(ren)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 128, 0), (0, 127, 0)])

    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report.colors_found, [True, False])


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_slider_widget():

    renderer = window.renderer()

    # Create 2 lines with 2 different colors
    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.array([[1., 0., 0.], [0.8, 0., 0.]])
    c = actor.line(lines, colors, linewidth=3)
    window.add(renderer, c)

    from dipy.viz.window import vtk

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    def print_status(obj, event):
        print(obj)
        print(event)
        renderer.SetBackground(np.random.rand(3))

    slider = widget.slider(iren=iren, callback=print_status)
    # text = widget.text(iren, None)

    iren.Initialize()

    ren_win.Render()
    iren.Start()
    arr = window.snapshot(renderer, size=(600, 600))
    report = window.analyze_snapshot(arr)


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_button_widget():

    from dipy.viz.window import vtk

    renderer = window.renderer()

    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.array([[1., 0., 0.], [0.8, 0., 0.]])
    stream_actor = actor.streamtube(lines, colors)

    window.add(renderer, stream_actor)

    renderer.ResetCamera()

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(600, 600)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    def callback(obj, event):
        print(obj)
        print('Pressed')

    fetch_viz_icons()
    button_png = read_viz_icons(fname='home3.png')

    button = widget.button(iren, callback,
                           button_png, (.8, 1.2), (50, 50))

    button_png_plus = read_viz_icons(fname='plus.png')
    button_plus = widget.button(iren, callback,
                                button_png_plus, (.7, .8), (50, 50))

    button_png_minus = read_viz_icons(fname='minus.png')
    button_minus = widget.button(iren, callback,
                                 button_png_minus, (.9, .8), (50, 50))

    def print_status(obj, event):
        print(obj)
        print(event)
        renderer.SetBackground(np.random.rand(3))

    slider = widget.slider(iren=iren, callback=print_status)

    iren.Initialize()

    ren_win.Render()

    button_norm_coords = (.9, 1.2)
    button_size = (50, 50)

    button.place(renderer)
    button_plus.place(renderer)
    button_minus.place(renderer)

    def win_callback(obj, event):
        # print(obj)
        print(event)
        print(obj.GetSize())

        button.place(renderer)
        button_plus.place(renderer)
        button_minus.place(renderer)

    ren_win.AddObserver(vtk.vtkCommand.ModifiedEvent, win_callback)

    ren_win.Render()
    iren.Start()

    arr = window.snapshot(renderer, size=(600, 600))


if __name__ == '__main__':

    test_window_module()
    # npt.run_module_suite()
