from typing import Any
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod

import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui

# Tab Types
SLICES_TAB = 'Slices Tab'
CLUSTERS_TAB = 'Clusters Tab'
ROIS_TAB = 'ROIs Tab'
PEAKS_TAB = 'Peaks Tab'


@dataclass
class HorizonUIElement:
    """
    Dataclass to define properties of horizon ui elements
    """
    visibility: bool
    selected_value: Any
    obj: Any

@dataclass
class HorizonSlider:
    """
    Dataclass to define HorizonSlider
    """
    label: HorizonUIElement
    slider: HorizonUIElement

class HorizonTab(ABC):
    tab_manager = None

    def __init__(self, tab_type=None):
        self._tab_type = tab_type

    @abstractmethod
    def build(self, tab_id, tab_ui, tab_manager):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def tab_type(self):
        """
        Type of the tab defined
        """
        return self._tab_type


class TabManager:
    def __init__(self, tabs, win_size, synchronize_slices=False):
        num_tabs = len(tabs)
        self._tabs = tabs
        self._synchronize_slices = synchronize_slices
        if not self._synchronize_slices:
            msg = 'Images are of different dimensions, ' \
                + 'synchronization of slices will not work'
            warnings.warn(msg)

        win_width, win_height = win_size

        # TODO: Add dynamic sizing
        # tab_size = (np.rint(win_width / 1.5), np.rint(win_height / 4.5))
        self.__tab_size = (1280, 240)
        x_pad = np.rint((win_width - self.__tab_size[0]) / 2)

        self.__tab_ui = ui.TabUI(
            position=(x_pad, 5), size=self.__tab_size, nb_tabs=num_tabs,
            active_color=(1, 1, 1), inactive_color=(0.5, 0.5, 0.5),
            draggable=True, startup_tab_id=0)

        for id, tab in enumerate(tabs):
            self.__tab_ui.tabs[id].title = ' ' + tab.name
            self.__tab_ui.tabs[id].title_font_size = 18
            tab.build(id, self.__tab_ui, self)

    def reposition(self, win_size):
        win_width, win_height = win_size
        x_pad = np.rint((win_width - self.__tab_size[0]) / 2)
        self.__tab_ui.position = (x_pad, 5)

    def synchronize_slices(self, active_tab_id, x_value, y_value, z_value):
        """
        Synchronize slicers for all the images

        Parameters
        ----------
        active_tab_id: int
            tab_id of the action performing tab
        x_value: float
            x-value of the active slicer
        y_value: float
            y-value of the active slicer
        z_value: float
            z-value of the active slicer
        """

        if not self._synchronize_slices:
            return

        slices_tabs = list(
            filter(
                lambda x: x.tab_type == SLICES_TAB
                and not x.tab_id == active_tab_id, self._tabs
            )
        )

        for slices_tab in slices_tabs:
            slices_tab.update_slices(x_value, y_value, z_value)

    @property
    def tab_ui(self):
        return self.__tab_ui


def build_label(text, font_size=16, bold=False):
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

def build_slider(
        initial_value,
        max_value,
        min_value=0,
        length=450,
        line_width=3,
        radius=8,
        font_size=16,
        text_template='{value:.1f} ({ratio:.0%})',
        on_moving_slider=lambda _slider: None,
        on_value_changed=lambda _slider: None,
        on_change=lambda _slider: None,
        label='',
        label_font_size=16,
        label_style_bold=False,
        is_double_slider=False
):
    """
    Creates a horizon theme based disk-knob slider

    Parameters
    ----------

    initial_value : float, (float, float)
        Initial value(s) of the slider.
    min_value : float
        Minimum value of the slider.
    max_value : float
        Maximum value of the slider.
    length : int
        Length of the slider.
    line_width : int
        Width of the line on which the disk will slide.
    radius : int
        Radius of the disk handle.
    font_size : int
        Size of the text to display alongside the slider (pt).
    text_template : str, callable
        If str, text template can contain one or multiple of the
        replacement fields: `{value:}`, `{ratio:}`.
        If callable, this instance of `:class:LineSlider2D` will be
        passed as argument to the text template function.
    on_moving_slider : callable
        When the slider is interacted by the user.
    on_value_changed : callable
        When value of the slider changed programmatically.
    on_change : callable
        When value of the slider changed.
    label : str
        Label to ui element for slider
    label_font_size : int
        Size of label text to display with slider
    label_style_bold : bool
        Is label should have bold style.

    Return
    ------

    {label: HorizonUIElement, slider: HorizonUIElement}
    """

    slider_label = build_label(
        label,
        font_size=label_font_size,
        bold=label_style_bold
    )


    # Initialize slider
    if not is_double_slider:
        slider = ui.LineSlider2D(
            initial_value=initial_value,
            max_value=max_value,
            min_value=min_value,
            length=length,
            line_width=line_width,
            outer_radius=radius,
            font_size=font_size,
            text_template=text_template
        )
    else:
        slider = ui.LineDoubleSlider2D(
        initial_values=initial_value,
        max_value=max_value,
        min_value=min_value,
        length=length,
        line_width=line_width,
        outer_radius=radius,
        font_size=font_size,
        text_template=text_template
    )

    # Attach callbacks
    slider.on_moving_slider = on_moving_slider
    slider.on_value_changed = on_value_changed
    slider.on_change = on_change

    # Apply color theme
    slider.default_color = (1., .5, .0)
    slider.track.color = (.8, .3, .0)
    slider.active_color = (.9, .4, .0)
    if not is_double_slider:
        slider.handle.color = (1., .5, .0)
    else:
        slider.handles[0].color = (1., .5, .0)
        slider.handles[1].color = (1., .5, .0)

    # Generate HorizonSlider
    return HorizonSlider(
        HorizonUIElement(True, label, slider_label),
        HorizonUIElement(True, initial_value, slider)
    )


def build_checkbox(
        labels=None,
        checked_labels=None,
        padding=1,
        font_size=16,
        on_change=lambda _checkbox: None
):
    """
    Creates horizon theme checkboxes

    Parameters
    ----------

    labels : list(str)
        List of labels of each option.
    checked_labels: list(str), optional
        List of labels that are checked on setting up.
    padding : float, optional
        The distance between two adjacent options
    font_size : int, optional
        Size of the text font.
    on_change : callback
        When checkbox value changed

    Returns
    -------

    checkbox : HorizonUIElement
    """

    if labels is None:
        labels = []

    if checked_labels is None:
        checked_labels = ()

    # Initialize checkboxes
    checkboxes = ui.Checkbox(
        labels=labels,
        checked_labels=checked_labels,
        padding=padding,
        font_size=font_size
    )

    # Attach callback
    checkboxes.on_change = on_change

    return HorizonUIElement(True, checked_labels, checkboxes)

def color_single_slider(slider):
    slider.default_color = (1., .5, .0)
    slider.track.color = (.8, .3, .0)
    slider.active_color = (.9, .4, .0)
    slider.handle.color = (1., .5, .0)


def color_double_slider(slider):
    slider.default_color = (1., .5, .0)
    slider.track.color = (.8, .3, .0)
    slider.active_color = (.9, .4, .0)
    slider.handles[0].color = (1., .5, .0)
    slider.handles[1].color = (1., .5, .0)
