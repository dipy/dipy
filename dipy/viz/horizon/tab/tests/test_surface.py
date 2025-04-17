from unittest.mock import patch

import pytest

from dipy.testing.decorators import set_random_number_generator, use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab.base import HorizonTab
from dipy.viz.horizon.tab.surface import SurfaceTab
from dipy.viz.horizon.visualizer.surface import SurfaceVisualizer

fury, has_fury, setup_module = optional_package("fury", min_version="0.10.0")
skip_it = use_xvfb == "skip" or not has_fury


@pytest.fixture
@set_random_number_generator()
def surface_viz(rng):
    """Fixture to create a Surface Visualizer."""
    vertices = rng.random((100, 3))
    faces = rng.integers(0, 100, size=(100, 3))

    class MockScene:
        def add(self, *actor):
            pass

    surface_viz = SurfaceVisualizer((vertices, faces), MockScene(), (0, 0, 0))

    return surface_viz


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_toggle_actors(surface_viz):
    """Test toggle_actors method in Surface Tab."""
    tab = SurfaceTab(surface_viz, "Surface 1", "mock_surface")
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
