"""Horizon tab UI components (deprecated)."""

from dipy.viz.horizon.tab.base import (
    HorizonTab,
    HorizonUIElement,
    TabManager,
    build_checkbox,
    build_label,
    build_radio_button,
    build_slider,
    build_switcher,
)
from dipy.viz.horizon.tab.cluster import ClustersTab
from dipy.viz.horizon.tab.peak import PeaksTab
from dipy.viz.horizon.tab.roi import ROIsTab
from dipy.viz.horizon.tab.slice import SlicesTab
from dipy.viz.horizon.tab.surface import SurfaceTab

__all__ = [
    "ClustersTab",
    "HorizonTab",
    "HorizonUIElement",
    "PeaksTab",
    "ROIsTab",
    "SlicesTab",
    "SurfaceTab",
    "TabManager",
    "build_checkbox",
    "build_label",
    "build_radio_button",
    "build_slider",
    "build_switcher",
]
