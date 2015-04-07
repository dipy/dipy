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
    # iren.Start()
    arr = window.snapshot(renderer, size=(600, 600))
    report = window.analyze_snapshot(renderer, arr)




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

    # button_png = '/home/eleftherios/Downloads/dipy-running-high-res.png'
    button_png = '/home/eleftherios/Devel/icons/icomoon/PNG/home3.png'
    # button_png = '/home/eleftherios/Devel/icons/antique-glowing-copper-orbs/antique-copper-orbs/antique-copper-orbs-netvibes-logo.png'
    button = widget.button(iren, callback,
                           button_png, (.8, 1.2), (50, 50))
    button_png_plus = '/home/eleftherios/Devel/icons/icomoon/PNG/plus.png'
    button_plus = widget.button(iren, callback,
                                button_png_plus, (.7, .8), (50, 50))
    button_png_minus = '/home/eleftherios/Devel/icons/icomoon/PNG/minus.png'
    button_minus = widget.button(iren, callback,
                                 button_png_minus, (.9, .8), (50, 50))


    from dipy.viz.widget import compute_bounds

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
    # iren.Start()

    arr = window.snapshot(renderer, size=(600, 600))

if __name__ == '__main__':

    # test_slider_widget()
    # test_button_widget()
    npt.run_module_suite()