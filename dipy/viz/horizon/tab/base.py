from abc import ABC, abstractmethod

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.gmem import GlobalHorizon

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class HorizonTab(ABC):
    @abstractmethod
    def build(self, tab_id, tab_ui):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass

class TabManager:
    def __init__(self, tabs, win_size):
        num_tabs = len(tabs)
        win_width, win_height = win_size
        
        tab_size = (np.rint(win_width / 1.5), np.rint(win_height / 4.5))
        x_pad = np.rint((win_width - tab_size[0]) / 2)
        
        self.__tab_ui = ui.TabUI(
            position=(x_pad, 5), size=tab_size, nb_tabs=num_tabs,
            active_color=(1, 1, 1), inactive_color=(0.5, 0.5, 0.5),
            draggable=False)
        
        #self.__tab_ui.parent_panel.color = (1., 1., 1.)
        #self.__tab_ui.parent_panel.opacity = 1.
        
        for id, tab in enumerate(tabs):
            self.__tab_ui.tabs[id].title = tab.name
            tab.build(id, self.__tab_ui)
            self.__tab_ui.tabs[id].color = (1., 1., 1.)
            #self.__tab_ui.tabs[id].opacity = .1
    
    @property
    def tab_ui(self):
        return self.__tab_ui


def build_label(text, font_size=18, bold=False):
    """
    Simple utility function to build labels

    Parameters
    ----------
    text : str
    font_size : int
    bold : bool

    Returns
    -------
    label : TextBlock2D
    """

    label = ui.TextBlock2D()
    label.message = text
    label.font_size = font_size
    label.font_family = 'Arial'
    label.justification = 'left'
    label.bold = bold
    label.italic = False
    label.shadow = False
    label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
    label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
    label.color = (0.7, 0.7, 0.7)

    return label


def color_single_slider(slider):
    slider.default_color = (1., .5, .0)
    slider.track.color = (.8, .3, .0)
    slider.active_color = (.9, .4, .0)
    slider.handle.color = (1., .5, .0)

def _color_dslider(slider):
    slider.default_color = (1, 0.5, 0)
    slider.track.color = (0.8, 0.3, 0)
    slider.active_color = (0.9, 0.4, 0)
    slider.handles[0].color = (1, 0.5, 0)
    slider.handles[1].color = (1, 0.5, 0)
