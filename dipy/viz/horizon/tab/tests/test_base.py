import numpy.testing as npt
import pytest

from dipy.testing.decorators import use_xvfb

from dipy.viz.horizon.tab import build_label

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab.base import color_double_slider, color_single_slider

fury, has_fury, setup_module = optional_package('fury')
if has_fury:
    from fury import ui

skip_it = use_xvfb == 'skip'

@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_build_label():
    # With empty string and default values of parameters
    generated_label = build_label("")
    npt.assert_equal(generated_label.message, "")
    npt.assert_equal(generated_label.font_size, 16)
    npt.assert_equal(generated_label.bold, False)
    npt.assert_equal(generated_label.font_family, 'Arial')
    npt.assert_equal(generated_label.justification, 'left')
    npt.assert_equal(generated_label.italic, False)
    npt.assert_equal(generated_label.shadow, False)
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundColor(), (0.0, 0.0, 0.0))
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundOpacity(), 0.0)
    npt.assert_equal(generated_label.color, (0.7, 0.7, 0.7))

    # With setting font_size
    generated_label = build_label("Hello", font_size=18)
    npt.assert_equal(generated_label.message, "Hello")
    npt.assert_equal(generated_label.font_size, 18)
    npt.assert_equal(generated_label.bold, False)
    npt.assert_equal(generated_label.font_family, 'Arial')
    npt.assert_equal(generated_label.justification, 'left')
    npt.assert_equal(generated_label.italic, False)
    npt.assert_equal(generated_label.shadow, False)
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundColor(), (0.0, 0.0, 0.0))
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundOpacity(), 0.0)
    npt.assert_equal(generated_label.color, (0.7, 0.7, 0.7))

    # With setting bold
    generated_label = build_label("Hello", bold=True)
    npt.assert_equal(generated_label.message, "Hello")
    npt.assert_equal(generated_label.font_size, 16)
    npt.assert_equal(generated_label.bold, True)
    npt.assert_equal(generated_label.font_family, 'Arial')
    npt.assert_equal(generated_label.justification, 'left')
    npt.assert_equal(generated_label.italic, False)
    npt.assert_equal(generated_label.shadow, False)
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundColor(), (0.0, 0.0, 0.0))
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundOpacity(), 0.0)
    npt.assert_equal(generated_label.color, (0.7, 0.7, 0.7))

    # With setting font_size and bold
    generated_label = build_label("Hello", font_size=20, bold=True)
    npt.assert_equal(generated_label.message, "Hello")
    npt.assert_equal(generated_label.font_size, 20)
    npt.assert_equal(generated_label.bold, True)
    npt.assert_equal(generated_label.font_family, 'Arial')
    npt.assert_equal(generated_label.justification, 'left')
    npt.assert_equal(generated_label.italic, False)
    npt.assert_equal(generated_label.shadow, False)
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundColor(), (0.0, 0.0, 0.0))
    npt.assert_equal(generated_label.actor.GetTextProperty().GetBackgroundOpacity(), 0.0)
    npt.assert_equal(generated_label.color, (0.7, 0.7, 0.7))


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_color_single_slider():
    line_slider = ui.LineSlider2D()

    color_single_slider(line_slider)
    npt.assert_equal(line_slider.default_color, (1., .5, .0))
    npt.assert_equal(line_slider.track.color, (.8, .3, .0))
    npt.assert_equal(line_slider.active_color, (.9, .4, .0))
    npt.assert_equal(line_slider.handle.color, (1., .5, .0))


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_color_double_slider():
    line_double_slider = ui.LineDoubleSlider2D()

    color_double_slider(line_double_slider)
    npt.assert_equal(line_double_slider.default_color, (1., .5, .0))
    npt.assert_equal(line_double_slider.track.color, (.8, .3, .0))
    npt.assert_equal(line_double_slider.active_color, (.9, .4, .0))
    npt.assert_equal(line_double_slider.handles[0].color, (1., .5, .0))
    npt.assert_equal(line_double_slider.handles[1].color, (1., .5, .0))