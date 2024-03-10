import pytest
import warnings

import numpy.testing as npt
from dipy.testing import check_for_warnings

from dipy.utils.optpkg import optional_package
from dipy.testing.decorators import use_xvfb
from dipy.viz.horizon.tab.base import (build_checkbox, build_label,
                                       build_slider, build_switcher)

fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")

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
    regular_label = build_label(text='Hello')
    npt.assert_equal(regular_label.message, 'Hello')
    npt.assert_equal(regular_label.font_size, 16)
    npt.assert_equal(regular_label.bold, False)
    check_label(regular_label)

    regular_label = build_label(text='Hello', font_size=10, bold=True)
    npt.assert_equal(regular_label.message, 'Hello')
    npt.assert_equal(regular_label.font_size, 10)
    npt.assert_equal(regular_label.bold, True)
    check_label(regular_label)

    horizon_label = build_label(text='Hello', is_horizon_label=True)
    npt.assert_equal(horizon_label.obj.message, 'Hello')
    npt.assert_equal(horizon_label.obj.font_size, 16)
    npt.assert_equal(horizon_label.obj.bold, False)
    check_label(horizon_label.obj)


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_build_slider():
    single_slider_label, single_slider = build_slider(5, 100)
    npt.assert_equal(single_slider_label.obj.message, '')
    npt.assert_equal(single_slider_label.obj.font_size, 16)
    npt.assert_equal(single_slider_label.obj.bold, False)
    npt.assert_equal(single_slider.obj.value, 5)
    npt.assert_equal(single_slider.obj.max_value, 100)
    npt.assert_equal(single_slider.obj.min_value, 0)
    npt.assert_equal(single_slider.obj.track.width, 450)
    npt.assert_equal(single_slider.obj.track.height, 3)
    npt.assert_equal(single_slider.obj.track.color, (.8, .3, .0))
    npt.assert_equal(single_slider.obj.default_color, (1., .5, .0))
    npt.assert_equal(single_slider.obj.active_color, (.9, .4, .0))
    npt.assert_equal(single_slider.obj.handle.color, (1., .5, .0))
    npt.assert_equal(single_slider.obj.handle.outer_radius, 8)
    npt.assert_equal(single_slider.obj.text.font_size, 16)
    npt.assert_equal(single_slider.obj.text_template,
                     '{value:.1f} ({ratio:.0%})')
    npt.assert_equal(single_slider.selected_value, 5)

    double_slider_label, double_slider = build_slider(
        (4, 5), 100, text_template='{value:.1f}', is_double_slider=True)
    npt.assert_equal(double_slider_label.obj.message, '')
    npt.assert_equal(double_slider_label.obj.font_size, 16)
    npt.assert_equal(double_slider_label.obj.bold, False)
    npt.assert_equal(double_slider.obj.max_value, 100)
    npt.assert_equal(double_slider.obj.min_value, 0)
    npt.assert_equal(double_slider.obj.track.width, 450)
    npt.assert_equal(double_slider.obj.track.height, 3)
    npt.assert_equal(double_slider.obj.track.color, (.8, .3, .0))
    npt.assert_equal(double_slider.obj.default_color, (1., .5, .0))
    npt.assert_equal(double_slider.obj.active_color, (.9, .4, .0))
    npt.assert_equal(double_slider.obj.handles[0].color, (1., .5, .0))
    npt.assert_equal(double_slider.obj.handles[1].color, (1., .5, .0))
    npt.assert_equal(double_slider.obj.handles[0].outer_radius, 8)
    npt.assert_equal(double_slider.obj.handles[1].outer_radius, 8)
    npt.assert_equal(double_slider.obj.text[0].font_size, 16)
    npt.assert_equal(double_slider.obj.text[1].font_size, 16)
    npt.assert_equal(double_slider.selected_value, (4, 5))


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_build_checkbox():
    checkbox = build_checkbox(['Hello', 'Hi'], ['Hello'])

    npt.assert_equal(len(checkbox.obj.checked_labels), 1)
    npt.assert_equal(len(checkbox.obj.labels), 2)
    npt.assert_equal(checkbox.selected_value, ['Hello'])

    with warnings.catch_warnings(record=True) as l_warns:
        checkbox = build_checkbox()
        npt.assert_equal(checkbox, None)
        check_for_warnings(l_warns, 'At least one label needs to be to create'
                           + ' checkboxes')

    with warnings.catch_warnings(record=True) as l_warns:
        checkbox = build_checkbox([])
        npt.assert_equal(checkbox, None)
        check_for_warnings(l_warns, 'At least one label needs to be to create'
                           + ' checkboxes')


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_build_switcher():
    with warnings.catch_warnings(record=True) as l_warns:
        none_switcher = build_switcher()
        npt.assert_equal(none_switcher, None)
        check_for_warnings(l_warns, 'No items passed in switcher')

    switcher_label, switcher = build_switcher(
        [{'label': 'Hello', 'value': 'hello'}])
    npt.assert_equal(switcher.selected_value[0], 0)
    npt.assert_equal(switcher.selected_value[1], 'hello')
    npt.assert_equal(switcher_label.selected_value, '')

    switcher_label, switcher = build_switcher(
        [{'label': 'Hello', 'value': 'hello'}], 'Greeting', 1)
    npt.assert_equal(switcher.selected_value[0], 0)
    npt.assert_equal(switcher.selected_value[1], 'hello')
    npt.assert_equal(switcher_label.selected_value, 'Greeting')
