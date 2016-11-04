import os
from collections import defaultdict

from os.path import join as pjoin

import numpy.testing as npt

from dipy.data import read_viz_icons, fetch_viz_icons
from dipy.viz import interactor
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
    renderer = window.ren()

    # Define some counter callback.
    states = defaultdict(lambda: 0)

    # Broken UI Element
    class BrokenUI(UI):

        def __init__(self):
            super(BrokenUI, self).__init__()

        def get_actors(self):
            pass

        def set_center(self, position):
            pass

    broken_ui = BrokenUI()
    npt.assert_raises(NotImplementedError, broken_ui.get_actors())
    npt.assert_raises(NotImplementedError, broken_ui.set_center((1, 2)))
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

    button_test.add_callback("RightButtonPressEvent", make_invisible)
    button_test.add_callback("LeftButtonPressEvent", modify_button_callback)
    # /Button

    current_size = (600, 600)
    show_manager = window.ShowManager(renderer, size=current_size, title="DIPY UI Example")

    renderer.add(button_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(states.items()))
    else:
        show_manager.play_events_from_file(recording_filename)
        msg = "Wrong count for '{}'."
        expected = [('CharEvent', 0),
                    ('KeyPressEvent', 0),
                    ('KeyReleaseEvent', 0),
                    ('MouseMoveEvent', 0),
                    ('LeftButtonPressEvent', 15),
                    ('RightButtonPressEvent', 1),
                    ('MiddleButtonPressEvent', 7),
                    ('LeftButtonReleaseEvent', 15),
                    ('MouseWheelForwardEvent', 0),
                    ('MouseWheelBackwardEvent', 0),
                    ('MiddleButtonReleaseEvent', 7),
                    ('RightButtonReleaseEvent', 1)]

        # Useful loop for debugging.
        for event, count in expected:
            if states[event] != count:
                print("{}: {} vs. {} (expected)".format(event,
                                                        states[event],
                                                        count))

        for event, count in expected:
            npt.assert_equal(states[event], count, err_msg=msg.format(event))
