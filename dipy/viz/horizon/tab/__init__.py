from dipy.viz.horizon.tab.base import (HorizonTab, TabManager, build_label,
                                       color_double_slider,
                                       color_single_slider)
from dipy.viz.horizon.tab.cluster import ClustersTab
from dipy.viz.horizon.tab.peak import PeaksTab
from dipy.viz.horizon.tab.roi import ROIsTab
from dipy.viz.horizon.tab.slice import SlicesTab

__all__ = [
    'HorizonTab', 'TabManager', 'ClustersTab', 'PeaksTab', 'ROIsTab',
    'SlicesTab', 'build_label', 'color_double_slider', 'color_single_slider']
