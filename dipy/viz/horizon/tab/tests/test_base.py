import pytest

import numpy.testing as npt

from dipy.utils.optpkg import optional_package
from dipy.testing.decorators import use_xvfb
from dipy.viz.horizon.tab.base import build_label, build_slider

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui

skip_it = use_xvfb == 'skip'

def check_label(label):
    npt.assert_equal(label.font_family, 'Arial')
    npt.assert_equal(label.justification, 'left')
    npt.assert_equal(label.italic, False)
    npt.assert_equal(label.shadow, False)
    npt.assert_equal(label.actor.GetTextProperty().GetBackgroundColor(),
                     (0., 0., 0.))
    npt.assert_equal(label.actor.GetTextProperty().GetBackgroundOpacity(), 0.0)
    npt.assert_equal(label.color, (0.7, 0.7, 0.7))

@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_build_label():
    # Regular label
    regular_label = build_label(text='Hello')
    npt.assert_equal(regular_label.message, 'Hello')
    npt.assert_equal(regular_label.font_size, 16)
    npt.assert_equal(regular_label.bold, False)
    check_label(regular_label)

    # Regular label with optional parameters
    regular_label = build_label(text='Hello', font_size=10, bold=True)
    npt.assert_equal(regular_label.message, 'Hello')
    npt.assert_equal(regular_label.font_size, 10)
    npt.assert_equal(regular_label.bold, True)
    check_label(regular_label)

    # HorizonUIElement
    horizon_label = build_label(text='Hello', is_horizon_label=True)
    npt.assert_equal(horizon_label.obj.message, 'Hello')
    npt.assert_equal(horizon_label.obj.font_size, 16)
    npt.assert_equal(horizon_label.obj.bold, False)
    check_label(horizon_label.obj)


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_build_slider():
    single_slider = build_slider(5, 100)
    npt.assert_equal(single_slider.label.obj.message, '')
    npt.assert_equal(single_slider.label.obj.font_size, 16)
    npt.assert_equal(single_slider.label.obj.bold, False)
    npt.assert_equal(single_slider.element.obj.value, 5)
    npt.assert_equal(single_slider.element.obj.max_value, 100)
    npt.assert_equal(single_slider.element.obj.min_value, 0)
    npt.assert_equal(single_slider.element.obj.track.width, 450)
    npt.assert_equal(single_slider.element.obj.track.height, 3)
    npt.assert_equal(single_slider.element.obj.track.color, (.8, .3, .0))
    npt.assert_equal(single_slider.element.obj.default_color, (1., .5, .0))
    npt.assert_equal(single_slider.element.obj.active_color, (.9, .4, .0))
    npt.assert_equal(single_slider.element.obj.handle.color, (1., .5, .0))
    npt.assert_equal(single_slider.element.obj.handle.outer_radius, 8)
    npt.assert_equal(single_slider.element.obj.text.font_size, 16)
    npt.assert_equal(single_slider.element.obj.text_template,
                     '{value:.1f} ({ratio:.0%})')
    npt.assert_equal(single_slider.element.selected_value, 5)

    double_slider = build_slider((4, 5), 100, text_template= '{value:.1f}',
                                 is_double_slider=True)
    npt.assert_equal(double_slider.label.obj.message, '')
    npt.assert_equal(double_slider.label.obj.font_size, 16)
    npt.assert_equal(double_slider.label.obj.bold, False)
    npt.assert_equal(double_slider.element.obj.max_value, 100)
    npt.assert_equal(double_slider.element.obj.min_value, 0)
    npt.assert_equal(double_slider.element.obj.track.width, 450)
    npt.assert_equal(double_slider.element.obj.track.height, 3)
    npt.assert_equal(double_slider.element.obj.track.color, (.8, .3, .0))
    npt.assert_equal(double_slider.element.obj.default_color, (1., .5, .0))
    npt.assert_equal(double_slider.element.obj.active_color, (.9, .4, .0))
    npt.assert_equal(double_slider.element.obj.handles[0].color, (1., .5, .0))
    npt.assert_equal(double_slider.element.obj.handles[1].color, (1., .5, .0))
    npt.assert_equal(double_slider.element.obj.handles[0].outer_radius, 8)
    npt.assert_equal(double_slider.element.obj.handles[1].outer_radius, 8)
    npt.assert_equal(double_slider.element.obj.text[0].font_size, 16)
    npt.assert_equal(double_slider.element.obj.text[1].font_size, 16)
    npt.assert_equal(double_slider.element.selected_value, (4, 5))
