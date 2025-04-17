from unittest.mock import patch

import numpy as np
import pytest

from dipy.direction.peaks import PeaksAndMetrics
from dipy.testing.decorators import set_random_number_generator, use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab.base import HorizonTab, HorizonUIElement
from dipy.viz.horizon.tab.peak import PeaksTab
from dipy.viz.horizon.visualizer.peak import PeaksVisualizer

fury, has_fury, setup_module = optional_package("fury", min_version="0.10.0")
skip_it = use_xvfb == "skip" or not has_fury


@pytest.fixture
@set_random_number_generator()
def peak_actor(rng):
    """Fixture to create a Peaks Actor."""
    peak_dirs = 255 * rng.random((5, 5, 5, 5, 3))
    pam = PeaksAndMetrics()
    pam.peak_dirs = peak_dirs
    pam.affine = np.eye(4)

    peak_viz = PeaksVisualizer((pam.peak_dirs, pam.affine), False, "mock_peaks")
    return peak_viz.actors[0]


@pytest.fixture
def slider():
    """Fixture to create a mock slider."""

    class MockSlider:
        def __init__(self):
            self.value = 0
            self.left_disk_value = 0
            self.right_disk_value = 100
            self.position = (0, 0)
            self.visibility = True

        def set_position(self, x, y):
            self.position = (x, y)

        def set_visibility(self, visibility):
            self.visibility = visibility

    return MockSlider()


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_init_params(peak_actor):
    """Test initialization of PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    assert isinstance(tab, PeaksTab)
    assert tab._actor == peak_actor
    assert tab.name == "Peaks 1"
    assert tab._file_name == "mock_peaks"


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_init_no_file_name(peak_actor):
    """Test initialization of PeaksTab without file name."""
    tab = PeaksTab(peak_actor, "Peaks 1", None)
    assert isinstance(tab, PeaksTab)
    assert tab._actor == peak_actor
    assert tab.name == "Peaks 1"
    assert tab._file_name == "Peaks 1"


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_init_ui_element_creation(peak_actor):
    """Test UI element creation in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    assert tab._opacity is not None
    assert tab._opacity_label is not None
    assert tab._slice_x is not None
    assert tab._slice_y is not None
    assert tab._slice_z is not None
    assert tab._slice_x_label is not None
    assert tab._slice_y_label is not None
    assert tab._slice_z_label is not None
    assert tab._slice_x_toggle is not None
    assert tab._slice_y_toggle is not None
    assert tab._slice_z_toggle is not None
    assert tab._range_x is not None
    assert tab._range_y is not None
    assert tab._range_z is not None
    assert tab._range_x_label is not None
    assert tab._range_y_label is not None
    assert tab._range_z_label is not None
    assert tab._view_mode_label is not None
    assert tab._view_mode_toggler is not None
    assert tab._file_label is not None
    assert tab._file_name_label is not None


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_update_ui_elements_position(peak_actor):
    """Test updating UI elements position in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")

    tab.build(0)

    assert tab._tab_id == 0

    x_pos = 0.02
    assert tab._actor_toggle.position == (x_pos, 0.85)
    assert tab._slice_x_toggle.position == (x_pos, 0.62)
    assert tab._slice_y_toggle.position == (x_pos, 0.38)
    assert tab._slice_z_toggle.position == (x_pos, 0.15)

    x_pos = 0.04
    assert tab._opacity_label.position == (x_pos, 0.85)
    assert tab._slice_x_label.position == (x_pos, 0.62)
    assert tab._slice_y_label.position == (x_pos, 0.38)
    assert tab._slice_z_label.position == (x_pos, 0.15)
    assert tab._range_x_label.position == (x_pos, 0.62)
    assert tab._range_y_label.position == (x_pos, 0.38)
    assert tab._range_z_label.position == (x_pos, 0.15)

    x_pos = 0.10
    assert tab._opacity.position == (x_pos, 0.85)
    assert tab._slice_x.position == (x_pos, 0.62)
    assert tab._slice_y.position == (x_pos, 0.38)
    assert tab._slice_z.position == (x_pos, 0.15)

    x_pos = 0.105
    assert tab._range_x.position == (x_pos, 0.66)
    assert tab._range_y.position == (x_pos, 0.42)
    assert tab._range_z.position == (x_pos, 0.19)

    x_pos = 0.52
    assert tab._view_mode_label.position == (x_pos, 0.85)
    assert tab._file_label.position == (x_pos, 0.28)

    x_pos = 0.62
    assert tab._view_mode_toggler.position == (x_pos, 0.80)
    assert tab._file_name_label.position == (x_pos, 0.28)


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_update_slices(peak_actor):
    """Test updating slices in PeaksTab when synced."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    tab.build(0)

    tab.update_slices(3, 2, 2)

    assert tab._slice_x.obj.value == 3
    assert tab._slice_y.obj.value == 2
    assert tab._slice_z.obj.value == 2


@patch.object(HorizonTab, "show")
@patch.object(HorizonTab, "hide")
@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_on_tab_selected(_m_show, _m_hide, peak_actor):
    """Test on_tab_selected method in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    tab.build(0)

    tab._view_mode_toggler.obj.checked_labels = ["Cross section"]

    with patch.object(PeaksTab, "_show_cross_section") as mock_show_cross_section:
        tab.on_tab_selected()
        mock_show_cross_section.assert_called_once()

    tab._view_mode_toggler.obj.checked_labels = ["Range"]

    with patch.object(PeaksTab, "_show_range") as mock_show_range:
        tab.on_tab_selected()
        mock_show_range.assert_called_once()


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_change_opcity(peak_actor, slider):
    """Test change_opacity method in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    tab.build(0)

    slider.value = 0.5
    tab._change_opacity(slider)
    assert tab._actor.global_opacity == 0.5


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_change_range(peak_actor, slider):
    """Test change_range method in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    tab.build(0)

    slider.left_disk_value = 10
    slider.right_disk_value = 90

    selected_range = HorizonUIElement(True, (0, 0), None)
    tab._change_range(slider, selected_range)

    assert selected_range.selected_value == (10, 90)


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_change_slice(peak_actor, slider):
    """Test change_slice method in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    tab.build(0)

    slider.value = 5
    selected_slice = HorizonUIElement(True, 0, None)
    tab._view_mode_toggler.obj.checked_labels = ["Cross section"]

    with patch.object(HorizonTab, "on_slice_change") as mock_sync:
        tab._change_slice(slider, selected_slice, sync_slice=True)
        mock_sync.assert_not_called()
        assert selected_slice.selected_value == 5

        slider.value = 10
        tab._change_slice(slider, selected_slice)
        mock_sync.assert_called_once()
        assert selected_slice.selected_value == 10


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_update_slice_visibility(peak_actor, slider):
    """Test update_slice_visibility method in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
    tab.build(0)

    selected_slice = HorizonUIElement(False, 0, slider)
    tab._update_slice_visibility(tab._slice_x_toggle.obj, selected_slice)
    assert selected_slice.visibility is True
    assert selected_slice.obj.visibility is True


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_toggle_actors(peak_actor):
    """Test toggle_actors method in PeaksTab."""
    tab = PeaksTab(peak_actor, "Peaks 1", "mock_peaks")
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
