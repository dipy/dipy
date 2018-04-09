import os
import numpy as np
from os.path import join as pjoin

from dipy.viz import actor, window, widget
from dipy.data import DATA_DIR
from dipy.data import fetch_viz_icons, read_viz_icons
import numpy.testing as npt
from dipy.testing.decorators import xvfb_it

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False


@npt.dec.skipif(not actor.have_vtk or not actor.have_vtk_colors or skip_it)
@xvfb_it
def test_button_and_slider_widgets():
    recording = False
    filename = "test_button_and_slider_widgets.log.gz"
    recording_filename = pjoin(DATA_DIR, filename)
    renderer = window.Renderer()

    # create some minimalistic streamlines
    lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
             np.array([[-1, 1, 0.], [1, 1, 0.]])]
    colors = np.array([[1., 0., 0.], [0.3, 0.7, 0.]])
    stream_actor = actor.streamtube(lines, colors)

    states = {'camera_button_count': 0,
              'plus_button_count': 0,
              'minus_button_count': 0,
              'slider_moved_count': 0,
              }

    renderer.add(stream_actor)

    # the show manager allows to break the rendering process
    # in steps so that the widgets can be added properly
    show_manager = window.ShowManager(renderer, size=(800, 800))

    if recording:
        show_manager.initialize()
        show_manager.render()

    def button_callback(obj, event):
        print('Camera pressed')
        states['camera_button_count'] += 1

    def button_plus_callback(obj, event):
        print('+ pressed')
        states['plus_button_count'] += 1

    def button_minus_callback(obj, event):
        print('- pressed')
        states['minus_button_count'] += 1

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
        states['slider_moved_count'] += 1

    slider = widget.slider(show_manager.iren, show_manager.ren,
                           callback=print_status,
                           min_value=-1,
                           max_value=1,
                           value=0.,
                           label="X",
                           right_normalized_pos=(.98, 0.6),
                           size=(120, 0), label_format="%0.2lf")

    # This callback is used to update the buttons/sliders' position
    # so they can stay on the right side of the window when the window
    # is being resized.

    global size
    size = renderer.GetSize()

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(states)
    else:
        show_manager.play_events_from_file(recording_filename)
        npt.assert_equal(states["camera_button_count"], 7)
        npt.assert_equal(states["plus_button_count"], 3)
        npt.assert_equal(states["minus_button_count"], 4)
        npt.assert_equal(states["slider_moved_count"], 116)

    if not recording:
        button.Off()
        slider.Off()
        # Uncomment below to test the slider and button with analyze
        # button.place(renderer)
        # slider.place(renderer)

        arr = window.snapshot(renderer, size=(800, 800))
        report = window.analyze_snapshot(arr)
        # import pylab as plt
        # plt.imshow(report.labels, origin='lower')
        # plt.show()
        npt.assert_equal(report.objects, 4)

    report = window.analyze_renderer(renderer)
    npt.assert_equal(report.actors, 1)


@npt.dec.skipif(not actor.have_vtk or not actor.have_vtk_colors or skip_it)
@xvfb_it
def test_text_widget():

    interactive = False

    renderer = window.Renderer()
    axes = actor.axes()
    window.add(renderer, axes)
    renderer.ResetCamera()

    show_manager = window.ShowManager(renderer, size=(900, 900))

    if interactive:
        show_manager.initialize()
        show_manager.render()

    fetch_viz_icons()
    button_png = read_viz_icons(fname='home3.png')

    def button_callback(obj, event):
        print('Button Pressed')

    button = widget.button(show_manager.iren,
                           show_manager.ren,
                           button_callback,
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

    if not interactive:
        button.Off()
        text.Off()
        pass

    if interactive:
        show_manager.render()
        show_manager.start()

    arr = window.snapshot(renderer, size=(900, 900))
    report = window.analyze_snapshot(arr)
    npt.assert_equal(report.objects, 3)

    # If you want to see the segmented objects after the analysis is finished
    # you can use imshow(report.labels, origin='lower')


if __name__ == '__main__':
    npt.run_module_suite()
