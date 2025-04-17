from unittest.mock import patch

import numpy as np
import pytest

from dipy.testing.decorators import set_random_number_generator, use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab.base import HorizonTab
from dipy.viz.horizon.tab.slice import SlicesTab
from dipy.viz.horizon.visualizer.slice import SlicesVisualizer

fury, has_fury, setup_module = optional_package("fury", min_version="0.10.0")
skip_it = use_xvfb == "skip" or not has_fury


@pytest.fixture
@set_random_number_generator()
def slice_viz(rng):
    """Fixture to create a Slice Actors."""
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -98.0],
            [0.0, 1.0, 0.0, -134.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    data = 255 * rng.random((197, 233, 189))

    class MockScene:
        def add(self, *actor):
            pass

    slice_viz = SlicesVisualizer(
        None, MockScene(), data, affine=affine, fname="mock_slices"
    )
    return slice_viz


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_toggle_actors(slice_viz):
    """Test toggle_actors method in SlicesTab."""
    tab = SlicesTab(slice_viz, "Slices 1", "mock_slices")
    tab.build(0)

    tab._actor_toggle.obj.checked_labels = [""]

    with patch.object(HorizonTab, "show") as mock_show:
        tab._toggle_actors(tab._actor_toggle.obj)
        mock_show.assert_called_with(*tab.actors)
        assert mock_show.call_count == 2

    tab._actor_toggle.obj.checked_labels = []

    with patch.object(HorizonTab, "hide") as mock_hide:
        tab._toggle_actors(tab._actor_toggle.obj)
        mock_hide.assert_called_with(*tab.actors)
        assert mock_hide.call_count == 2
