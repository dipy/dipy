import numpy as np

from dipy.viz import actor
from dipy.viz import window
from dipy.viz import widget

import numpy.testing as npt

@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_slider_widget():

    renderer = window.renderer()

    # Create 2 lines with 2 different colors
    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.array([[1., 0., 0.], [0.8, 0., 0.]])
    c = actor.streamtube(lines, colors)
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

    iren.Initialize()

    ren_win.Render()
    iren.Start()


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

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    def callback(obj, event):
        print(obj)
        print('Pressed')

    button_png = '/home/eleftherios/Downloads/dipy-running-high-res.png'
    # button_png = '/home/eleftherios/Devel/icons/antique-glowing-copper-orbs/antique-copper-orbs/antique-copper-orbs-netvibes-logo.png'
    button = widget.button(iren=iren, callback=callback,
                           fname=button_png)

    from dipy.viz.widget import compute_bounds

    iren.Initialize()

    ren_win.Render()

    button_norm_coords = (.9, 1.2)
    button_size = (100, 100)

    bds = compute_bounds(renderer, button_norm_coords, button_size)
    button.GetRepresentation().PlaceWidget(bds)
    button.On()

    def win_callback(obj, event):
        # print(obj)
        print(event)
        print(obj.GetSize())

        bds = compute_bounds(renderer, button_norm_coords, button_size)
        button.GetRepresentation().PlaceWidget(bds)
        button.On()

    ren_win.AddObserver(vtk.vtkCommand.ModifiedEvent, win_callback)

    ren_win.Render()

    iren.Start()


if __name__ == '__main__':

    #test_slider_widget()
    test_button_widget()
