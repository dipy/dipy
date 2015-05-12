import numpy as np
from dipy.viz import actor, window, widget, fvtk
from dipy.data import fetch_viz_icons, read_viz_icons
import numpy.testing as npt


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_button_and_slider_widgets():

    renderer = window.Renderer()

    # create some random streamlines
    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.array([[1., 0., 0.], [0.3, 0.7, 0.]])
    stream_actor = actor.streamtube(lines, colors)

    renderer.add(stream_actor)

    # the show manager allows to break the rendering process
    # in steps so that the widgets can be added properly
    show_manager = window.ShowManager(renderer, size=(800, 800))

    def callback(obj, event):
        print(obj)
        print('Pressed')

    fetch_viz_icons()
    button_png = read_viz_icons(fname='camera.png')

    button = widget.button(show_manager.iren, callback,
                           button_png, (.98, 1.), (80, 50))

    button_png_plus = read_viz_icons(fname='plus.png')
    button_plus = widget.button(show_manager.iren, callback,
                                button_png_plus, (.98, .9), (120, 50))

    button_png_minus = read_viz_icons(fname='minus.png')
    button_minus = widget.button(show_manager.iren, callback,
                                 button_png_minus, (.98, .9), (50, 50))

    def print_status(obj, event):
        # print(obj)
        # print(event)
        renderer.SetBackground(np.random.rand(3))

    slider = widget.slider(show_manager.iren, show_manager.ren,
                           callback=print_status,
                           right_normalized_pos=(.98, 0.7),
                           size=(120, 0))

    show_manager.initialize()
    show_manager.render()

    button.place(renderer)
    button_plus.place(renderer)
    button_minus.place(renderer)
    slider.place(renderer)

    def win_callback(obj, event):
        # print(obj)
        print(event)
        print(obj.GetSize())

        button.place(renderer)
        button_plus.place(renderer)
        button_minus.place(renderer)
        slider.place(renderer)

    # ren_win.AddObserver(vtk.vtkCommand.ModifiedEvent, win_callback)
    show_manager.add_window_callback(win_callback)

    # show_manager.render()

    show_manager.start()

    arr = window.snapshot(renderer, size=(800, 800))


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_text_widget():

    interactive = False

    renderer = window.Renderer()

    axes = fvtk.axes()

    window.add(renderer, axes)

    renderer.ResetCamera()

    show_manager = window.ShowManager(renderer, size=(1200, 1200))

    if interactive:
        show_manager.initialize()
        show_manager.render()

    fetch_viz_icons()
    button_png = read_viz_icons(fname='home3.png')

    def button_callback(obj, event):
        print('Button Pressed')

    button = widget.button(show_manager.iren, button_callback,
                           button_png, (.8, 1.2), (100, 100))

    global rulez
    rulez = True

    def text_callback(obj, event):

        global rulez
        print('Text selected')
        if rulez:
            obj.GetTextActor().SetInput("Diffusion Imaging Rulez!!")
            rulez = False
        else:
            obj.GetTextActor().SetInput("Diffusion Imaging in Python")
            rulez = True
        show_manager.render()

    text = widget.text(show_manager.iren,
                       show_manager.ren,
                       text_callback,
                       message="Diffusion Imaging in Python",
                       left_down_pos=(0., 0.),
                       right_top_pos=(0.4, 0.05),
                       opacity=1.,
                       border=False)

    button.place(renderer)
    text.place(renderer)

    if interactive:
        show_manager.render()

    def win_callback(obj, event):
        print('Window modified')
        button.place(renderer)
        text.place(renderer)

    if interactive:
        show_manager.add_window_callback(win_callback)
        show_manager.render()
        show_manager.start()

    arr = window.snapshot(renderer, size=(1200, 1200))
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 30)

    # imshow(report.labels, origin='lower')


if __name__ == '__main__':

    npt.run_module_suite()