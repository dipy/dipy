from abc import ABC, abstractmethod

from dipy.utils.optpkg import optional_package
from dipy.viz.gmem import GlobalHorizon

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class HorizonTab(ABC):
    @abstractmethod
    def build(self, tab_id, tab_ui):
        pass
    
    @abstractmethod
    @property
    def name(self):
        pass

class TabManager:
    def __init__(self, tabs):
        # TODO: Handle size dynamically
        num_tabs = len(tabs)
        
        self.__tab_ui = ui.TabUI(
            position=(-25, 5), size=(320, 240), nb_tabs=num_tabs,
            active_color=(1, 1, 1), inactive_color=(0.5, 0.5, 0.5),
            draggable=False
        )
        
        for id, tab in enumerate(tabs):
            self.__tab_ui.tabs[id].title = tab.name
            tab.build(id, self.__tab_ui)
    
    @property
    def tab_ui(self):
        return self.__tab_ui
