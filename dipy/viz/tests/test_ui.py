import os
import sys
import pickle

from os.path import join as pjoin
import numpy.testing as npt

from dipy.data import read_viz_icons, fetch_viz_icons
from dipy.viz import ui
from dipy.viz import window
from dipy.data import DATA_DIR

from dipy.viz.ui import UI

from dipy.testing.decorators import xvfb_it

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False

if have_vtk:
    print("Using VTK {}".format(vtk.vtkVersion.GetVTKVersion()))


class EventCounter(object):
    def __init__(self, events_names=["CharEvent",
                                     "MouseMoveEvent",
                                     "KeyPressEvent",
                                     "KeyReleaseEvent",
                                     "LeftButtonPressEvent",
                                     "LeftButtonReleaseEvent",
                                     "RightButtonPressEvent",
                                     "RightButtonReleaseEvent",
                                     "MiddleButtonPressEvent",
                                     "MiddleButtonReleaseEvent"]):
        # Events to count
        self.events_counts = {name: 0 for name in events_names}

    def count(self, i_ren, obj, element):
        """ Simple callback that counts events occurences. """
        self.events_counts[i_ren.event.name] += 1

    def monitor(self, ui_component):
        for event in self.events_counts:
            for actor in ui_component.get_actors():
                ui_component.add_callback(actor, event, self.count)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.events_counts, f, protocol=-1)

    @classmethod
    def load(cls, filename):
        event_counter = cls()
        with open(filename, 'rb') as f:
            event_counter.events_counts = pickle.load(f)

        return event_counter

    def check_counts(self, expected):
        npt.assert_equal(len(self.events_counts),
                         len(expected.events_counts))

        # Useful loop for debugging.
        msg = "{}: {} vs. {} (expected)"
        for event, count in expected.events_counts.items():
            if self.events_counts[event] != count:
                print(msg.format(event, self.events_counts[event], count))

        msg = "Wrong count for '{}'."
        for event, count in expected.events_counts.items():
            npt.assert_equal(self.events_counts[event], count,
                             err_msg=msg.format(event))


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_broken_ui_component():
    class BrokenUI(UI):
        def __init__(self):
            self.actor = vtk.vtkActor()
            super(BrokenUI, self).__init__()

    broken_ui = BrokenUI()
    npt.assert_raises(NotImplementedError, broken_ui.get_actors)
    npt.assert_raises(NotImplementedError, broken_ui.set_center, (1, 2))


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_wrong_interactor_style():
    panel = ui.Panel2D(center=(440, 90), size=(300, 150))
    dummy_renderer = window.Renderer()
    dummy_show_manager = window.ShowManager(dummy_renderer,
                                            interactor_style='trackball')
    npt.assert_raises(TypeError, panel.add_to_renderer, dummy_renderer)


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_ui_button_panel(recording=False):
    filename = "test_ui_button_panel"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".pkl")

    # Rectangle
    rectangle_test = ui.Rectangle2D(size=(10, 10))
    rectangle_test.get_actors()
    another_rectangle_test = ui.Rectangle2D(size=(1, 1))
    # /Rectangle

    # Button
    fetch_viz_icons()

    icon_files = dict()
    icon_files['stop'] = read_viz_icons(fname='stop2.png')
    icon_files['play'] = read_viz_icons(fname='play3.png')

    button_test = ui.Button2D(icon_fnames=icon_files)
    button_test.set_center((20, 20))

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
    panel = ui.Panel2D(center=(440, 90), size=(300, 150),
                       color=(1, 1, 1), align="right")
    panel.add_element(rectangle_test, 'absolute', (580, 150))
    panel.add_element(button_test, 'relative', (0.2, 0.2))
    npt.assert_raises(ValueError, panel.add_element, another_rectangle_test,
                      'error_string', (1, 2))
    # /Panel

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(button_test)
    event_counter.monitor(panel)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title="DIPY Button")

    show_manager.ren.add(panel)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_ui_textbox(recording=False):
    filename = "test_ui_textbox"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".pkl")

    # TextBox
    textbox_test = ui.TextBox2D(height=3, width=10, text="Text")

    another_textbox_test = ui.TextBox2D(height=3, width=10, text="Enter Text")
    another_textbox_test.set_message("Enter Text")
    another_textbox_test.set_center((10, 100))
    # /TextBox

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(textbox_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size, title="DIPY TextBox")

    show_manager.ren.add(textbox_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_text_actor_2d():
    # TextActor2D
    text_actor = ui.TextActor2D()
    text_actor.message = "Hello World!"
    npt.assert_equal("Hello World!", text_actor.message)
    text_actor.font_size = 18
    npt.assert_equal("18", str(text_actor.font_size))
    text_actor.font_family = "Arial"
    npt.assert_equal("Arial", text_actor.font_family)
    with npt.assert_raises(ValueError):
        text_actor.font_family = "Verdana"
    text_actor.justification = "left"
    text_actor.justification = "right"
    text_actor.justification = "center"
    npt.assert_equal("Centered", text_actor.justification)
    with npt.assert_raises(ValueError):
        text_actor.justification = "bottom"
    text_actor.bold = True
    text_actor.bold = False
    npt.assert_equal(False, text_actor.bold)
    text_actor.italic = True
    text_actor.italic = False
    npt.assert_equal(False, text_actor.italic)
    text_actor.shadow = True
    text_actor.shadow = False
    npt.assert_equal(False, text_actor.shadow)
    text_actor.color = (1, 0, 0)
    npt.assert_equal((1, 0, 0), text_actor.color)
    text_actor.position = (2, 3)
    npt.assert_equal((2, 3), text_actor.position)
    # /TextActor2D


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_ui_line_slider_2d(recording=False):
    filename = "test_ui_line_slider_2d"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".pkl")

    line_slider_2d_test = ui.LineSlider2D(initial_value=-2,
                                          min_value=-5, max_value=5)
    line_slider_2d_test.set_center((300, 300))

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(line_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size,
                                      title="DIPY Line Slider")

    show_manager.ren.add(line_slider_2d_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


@npt.dec.skipif(not have_vtk or skip_it)
@xvfb_it
def test_ui_disk_slider_2d(recording=False):
    filename = "test_ui_disk_slider_2d"
    recording_filename = pjoin(DATA_DIR, filename + ".log.gz")
    expected_events_counts_filename = pjoin(DATA_DIR, filename + ".pkl")

    disk_slider_2d_test = ui.DiskSlider2D()
    disk_slider_2d_test.set_center((300, 300))
    disk_slider_2d_test.value = 90

    # Assign the counter callback to every possible event.
    event_counter = EventCounter()
    event_counter.monitor(disk_slider_2d_test)

    current_size = (600, 600)
    show_manager = window.ShowManager(size=current_size,
                                      title="DIPY Disk Slider")

    show_manager.ren.add(disk_slider_2d_test)

    if recording:
        show_manager.record_events_to_file(recording_filename)
        print(list(event_counter.events_counts.items()))
        event_counter.save(expected_events_counts_filename)

    else:
        show_manager.play_events_from_file(recording_filename)
        expected = EventCounter.load(expected_events_counts_filename)
        event_counter.check_counts(expected)


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "test_ui_button_panel":
        test_ui_button_panel(recording=True)

    if len(sys.argv) <= 1 or sys.argv[1] == "test_ui_textbox":
        test_ui_textbox(recording=True)

    if len(sys.argv) <= 1 or sys.argv[1] == "test_ui_line_slider_2d":
        test_ui_line_slider_2d(recording=True)

    if len(sys.argv) <= 1 or sys.argv[1] == "test_ui_disk_slider_2d":
        test_ui_disk_slider_2d(recording=True)
