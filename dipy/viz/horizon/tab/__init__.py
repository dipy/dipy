from dipy.viz.horizon.tab.base import (HorizonTab, TabManager, build_label,
                                       build_slider, build_checkbox,
                                       build_switcher, color_double_slider,
                                       color_single_slider, build_radio_button)
from dipy.viz.horizon.tab.cluster import ClustersTab
from dipy.viz.horizon.tab.peak import PeaksTab
from dipy.viz.horizon.tab.roi import ROIsTab
from dipy.viz.horizon.tab.slice import SlicesTab
from dipy.viz.horizon.tab.surface import SurfaceTab

__all__ = [
    'HorizonTab', 'TabManager', 'ClustersTab', 'PeaksTab', 'ROIsTab',
    'SlicesTab', 'build_label', 'build_slider', 'build_checkbox',
    'build_switcher', 'SurfaceTab', 'build_radio_button'
    'color_double_slider', 'color_single_slider']
