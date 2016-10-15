import os
import numpy as np
from os.path import join as pjoin
from collections import defaultdict

from dipy.viz import actor, window, interactor
from dipy.data import DATA_DIR
import numpy.testing as npt
from dipy.testing.decorators import xvfb_it

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False


@npt.dec.skipif(not actor.have_vtk or not actor.have_vtk_colors or skip_it)
@xvfb_it
def test_custom_interactor_style_events():

    recording = False
    recording_filename = pjoin(DATA_DIR, "test_custom_interactor_style_events.log.gz")
    renderer = window.Renderer()

    # the show manager allows to break the rendering process
    # in steps so that the widgets can be added properly
    interactor_style = interactor.CustomInteractorStyle()
    show_manager = window.ShowManager(renderer, size=(800, 800), interactor_style=interactor_style)

    # create some minimalistic streamlines
    lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
             np.array([[-1, 1, 0.], [1, 1, 0.]])]
    colors = np.array([[1., 0., 0.], [0.3, 0.7, 0.]])
    tube1 = actor.streamtube([lines[0]], colors[0])
    tube2 = actor.streamtube([lines[1]], colors[1])
    # renderer.add(stream_actor)
    renderer.add(tube1)
    renderer.add(tube2)

    # Define some counter callback.
    states = defaultdict(lambda: 0)
    def counter(obj, event):
        states[event] += 1

    # Assign the counter callback to every possible event.
    for event in ["CharEvent", "MouseMoveEvent",
                  "KeyPressEvent", "KeyReleaseEvent",
                  "LeftButtonPressEvent", "LeftButtonReleaseEvent",
                  "RightButtonPressEvent", "RightButtonReleaseEvent",
                  "MiddleButtonPressEvent", "MiddleButtonReleaseEvent",
                  "MouseWheelForwardEvent", "MouseWheelBackwardEvent"]:
        interactor.add_callback(tube1, event, counter)

    # Add callback to scale up/down tube1.
    def scale_up_obj(obj, event):
        scale = np.array(obj.GetScale()) + 0.1
        obj.SetScale(*scale)
        show_manager.render()
        return True  # Stop propagating the event.

    def scale_down_obj(obj, event):
        scale = np.array(obj.GetScale()) - 0.1
        obj.SetScale(*scale)
        show_manager.render()
        return True  # Stop propagating the event.

    interactor.add_callback(tube2, "MouseWheelForwardEvent", scale_up_obj)
    interactor.add_callback(tube2, "MouseWheelBackwardEvent", scale_down_obj)

    # Add callback to hide/show tube1.
    def toggle_visibility(obj, event):
        key = show_manager.iren.GetInteractorStyle().GetKeySym()
        if key.lower() == "v":
            obj.SetVisibility(not obj.GetVisibility())
            show_manager.render()

    show_manager.iren.GetInteractorStyle().add_active_prop(tube1)
    show_manager.iren.GetInteractorStyle().add_active_prop(tube2)
    show_manager.iren.GetInteractorStyle().remove_active_prop(tube2)
    interactor.add_callback(tube1, "CharEvent", toggle_visibility)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(states.items()))
    else:
        show_manager.play_events_from_file(recording_filename)
        npt.assert_equal(states['CharEvent'], 2)
        npt.assert_equal(states['KeyPressEvent'], 2)
        npt.assert_equal(states['KeyReleaseEvent'], 2)
        npt.assert_equal(states['MouseMoveEvent'], 1014)
        npt.assert_equal(states['LeftButtonPressEvent'], 1)
        npt.assert_equal(states['RightButtonPressEvent'], 1)
        npt.assert_equal(states['MiddleButtonPressEvent'], 1)
        npt.assert_equal(states['LeftButtonReleaseEvent'], 1)
        npt.assert_equal(states['RightButtonReleaseEvent'], 1)
        npt.assert_equal(states['MiddleButtonReleaseEvent'], 1)
        npt.assert_equal(states['MouseWheelForwardEvent'], 17)
        npt.assert_equal(states['MouseWheelBackwardEvent'], 20)
