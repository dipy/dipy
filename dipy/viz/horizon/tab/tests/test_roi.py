from unittest.mock import patch

import numpy as np
import pytest

from dipy.testing.decorators import use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab.base import HorizonTab
from dipy.viz.horizon.tab.roi import ROIsTab

fury, has_fury, setup_module = optional_package("fury", min_version="0.10.0")
skip_it = use_xvfb == "skip" or not has_fury

if has_fury:
    from fury.actor import contour_from_roi


@pytest.fixture
def roi_actors():
    """Fixture to create a ROI Actors."""
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -98.0],
            [0.0, 1.0, 0.0, -134.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = np.zeros((197, 233, 189))
    img[0:25, :, :] = 1

    roi_actor = contour_from_roi(img, affine=affine)
    return [roi_actor]


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_toggle_actors(roi_actors):
    """Test toggle_actors method in ROIsTab."""
    tab = ROIsTab(roi_actors)
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
