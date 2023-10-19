import pytest

from dipy.utils.optpkg import optional_package
from dipy.testing.decorators import use_xvfb
from dipy.viz.horizon.tab.base import build_label

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui

skip_it = use_xvfb == 'skip'

# @pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
# def test_build_label():
#     # Regular label
#     regular_label = build_label(text)