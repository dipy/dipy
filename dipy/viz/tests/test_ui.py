import os
from collections import defaultdict

from os.path import join as pjoin

import numpy.testing as npt

from dipy.data import read_viz_icons, fetch_viz_icons
from dipy.viz import ui
from dipy.viz import window
from dipy.data import DATA_DIR

from dipy.testing.decorators import xvfb_it

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
from dipy.viz.ui import UI

vtk, have_vtk, setup_module = optional_package('vtk')

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_button(recording=False):
    print("Using VTK {}".format(vtk.vtkVersion.GetVTKVersion()))
    filename = "test_ui.log.gz"
    recording_filename = pjoin(DATA_DIR, filename)

    # Define some counter callback.
    states = defaultdict(lambda: 0)

    # Broken UI Element
    class BrokenUI(UI):

        def __init__(self):
            self.actor = vtk.vtkActor()
            super(BrokenUI, self).__init__()

        def add_callback(self, event_type, callback):
            """ Adds events to an actor.
            Parameters
            ----------
            event_type : string
                event code
            callback : function
                callback function
            """
            super(BrokenUI, self).add_callback(self.actor, event_type, callback)

    broken_ui = BrokenUI()
    npt.assert_raises(NotImplementedError, broken_ui.get_actors)
    npt.assert_raises(NotImplementedError, broken_ui.set_center, (1, 2))
    # /Broken UI Element

    # Button
    fetch_viz_icons()

    icon_files = dict()
    icon_files['stop'] = read_viz_icons(fname='stop2.png')
    icon_files['play'] = read_viz_icons(fname='play3.png')

    button_test = ui.Button2D(icon_fnames=icon_files)
    button_test.set_center((20, 20))

    def counter(i_ren, obj, button):
        states[i_ren.event.name] += 1

    # Assign the counter callback to every possible event.
    for event in ["CharEvent", "MouseMoveEvent",
                  "KeyPressEvent", "KeyReleaseEvent",
                  "LeftButtonPressEvent", "LeftButtonReleaseEvent",
                  "RightButtonPressEvent", "RightButtonReleaseEvent",
                  "MiddleButtonPressEvent", "MiddleButtonReleaseEvent"]:
        button_test.add_callback(event, counter)

    def make_invisible(i_ren, obj, button):
        # i_ren: CustomInteractorStyle
        # obj: vtkActor picked
        # button: Button2D
        button.set_visibility(False)
        i_ren.force_render()
        i_ren.event.abort()

    def modify_button_callback(i_ren, obj, button):
        # i_ren: CustomInteractorStyle
        # obj: vtkActor picked
        # button: Button2D
        button.next_icon()
        i_ren.force_render()

    button_test.on_right_mouse_button_pressed = make_invisible
    button_test.on_left_mouse_button_pressed = modify_button_callback

    button_test.scale((2, 2))
    button_color = button_test.color
    button_test.color = button_color
    # /Button

    # Panel
    panel = ui.Panel2D(center=(440, 90), size=(300, 150), color=(1, 1, 1), align="right")
    panel.add_element(button_test, 'relative', (0.2, 0.2))
    # /Panel

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title="DIPY UI Example")

    show_manager.ren.add(panel)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(states.items()))
    else:
        show_manager.play_events_from_file(recording_filename)
        msg = "Wrong count for '{}'."
        expected = [('CharEvent', 0),
                    ('KeyPressEvent', 0),
                    ('KeyReleaseEvent', 0),
                    ('MouseMoveEvent', 161),
                    ('LeftButtonPressEvent', 12),
                    ('RightButtonPressEvent', 3),
                    ('MiddleButtonPressEvent', 0),
                    ('LeftButtonReleaseEvent', 12),
                    ('MouseWheelForwardEvent', 0),
                    ('MouseWheelBackwardEvent', 0),
                    ('MiddleButtonReleaseEvent', 0),
                    ('RightButtonReleaseEvent', 3)]

        # Useful loop for debugging.
        for event, count in expected:
            if states[event] != count:
                print("{}: {} vs. {} (expected)".format(event,
                                                        states[event],
                                                        count))

        for event, count in expected:
            npt.assert_equal(states[event], count, err_msg=msg.format(event))

            # Dummy Show Manager
        dummy_renderer = window.Renderer()
        dummy_show_manager = window.ShowManager(dummy_renderer, size=(800, 800), reset_camera=False,
                                                interactor_style='trackball')
        npt.assert_raises(TypeError, button_test.add_to_renderer, dummy_renderer)
        # /Dummy Show Manager

