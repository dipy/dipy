"""
=====================================
Create a minimalistic user interface
=====================================

"""

from dipy.viz import window, actor, widget
from dipy.data import fetch_viz_icons, read_viz_icons

interactive = True
renderer = window.Renderer()


lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
         np.array([[-1, 1, 0.], [1, 1, 0.]])]
colors = np.array([[1., 0., 0.], [0.3, 0.7, 0.]])
stream_actor = actor.streamtube(lines, colors)

renderer.add(stream_actor)

# the show manager allows to break the rendering process
# in steps so that the widgets can be added properly
show_manager = window.ShowManager(renderer, size=(800, 800))

if interactive:
    show_manager.initialize()
    show_manager.render()


global opacity
opacity = 1.

def button_callback(obj, event):
    print('Camera pressed')
    window.save_file_dialog(file_types=[("PNG files", "*.png")])

def button_plus_callback(obj, event):
    print('+ pressed')
    global opacity
    if opacity < 1:
        opacity += 0.2
    stream_actor.GetProperty().SetOpacity(opacity)


def button_minus_callback(obj, event):
    print('- pressed')
    global opacity
    if opacity > 0:
        opacity -= 0.2

    stream_actor.GetProperty().SetOpacity(opacity)

fetch_viz_icons()
button_png = read_viz_icons(fname='camera.png')

button = widget.button(show_manager.iren,
                       show_manager.ren,
                       button_callback,
                       button_png, (.98, 1.), (80, 50))

button_png_plus = read_viz_icons(fname='plus.png')
button_plus = widget.button(show_manager.iren,
                            show_manager.ren,
                            button_plus_callback,
                            button_png_plus, (.98, .9), (120, 50))

button_png_minus = read_viz_icons(fname='minus.png')
button_minus = widget.button(show_manager.iren,
                             show_manager.ren,
                             button_minus_callback,
                             button_png_minus, (.98, .9), (50, 50))

def print_status(obj, event):
    rep = obj.GetRepresentation()
    stream_actor.SetPosition((rep.GetValue(), 0, 0))

slider = widget.slider(show_manager.iren, show_manager.ren,
                       callback=print_status,
                       min_value=-1,
                       max_value=1,
                       value=0.,
                       label="X",
                       right_normalized_pos=(.98, 0.7),
                       size=(120, 0), label_format="%0.2lf")

# This callback is used to update the buttons/sliders' position
# so they can stay on the right side of the window when the window
# is being resized.

global size
size = renderer.GetSize()

def win_callback(obj, event):
    global size
    if size != obj.GetSize():

        button.place(renderer)
        button_plus.place(renderer)
        button_minus.place(renderer)
        slider.place(renderer)
        size = obj.GetSize()
        show_manager.render()

if interactive:
    show_manager.add_window_callback(win_callback)
    # you can also register any callback in a vtk way like this
    # show_manager.window.AddObserver(vtk.vtkCommand.ModifiedEvent,
    #                                 win_callback)

    show_manager.render()
    show_manager.start()

if not interactive:
    button.Off()
    slider.Off()

    arr = window.snapshot(renderer, size=(800, 800))
    # imshow(report.labels, origin='lower')

report = window.analyze_renderer(renderer)
