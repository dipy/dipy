from typing import Any
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod

import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui
    from fury.data import read_viz_icons


@dataclass
class HorizonUIElement:
    """
    Dataclass to define properties of horizon ui elements
    """
    visibility: bool
    selected_value: Any
    obj: Any
    position = (0, 0)

@dataclass
class HorizonCombineElement:
    """
    Dataclass to define HorizonSlider
    """
    label: HorizonUIElement
    element: HorizonUIElement

class HorizonTab(ABC):

    def __init__(self):
        self._elements = []

    @abstractmethod
    def build(self, tab_id, tab_ui):
        pass

    def register_elements(self, elements):
        """
        Register elements for rendering
        """
        if hasattr(elements, 'label'):
            self._elements.append(elements.label)
            self._elements.append(elements.element)
        else:
            self._elements.append(elements)

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def tab_type(self):
        pass

    @property
    def elements(self):
        """
        list of underlying FURY ui elements in the tab.
        """
        return self._elements

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

        self._tab_ui = ui.TabUI(
            position=(x_pad, 5), size=self.__tab_size, nb_tabs=num_tabs,
            active_color=(1, 1, 1), inactive_color=(0.5, 0.5, 0.5),
            draggable=True, startup_tab_id=0)

        for tab_id, tab in enumerate(tabs):
            self._tab_ui.tabs[tab_id].title = ' ' + tab.name
            self._tab_ui.tabs[tab_id].title_font_size = 18
            tab.build(tab_id, self._tab_ui)
            if tab.tab_type == 'slices_tab':
                tab.on_slice_change = self.synchronize_slices
                self._render_tab_elements(tab_id, tab.elements)

    def _render_tab_elements(self, tab_id, elements):
        print(elements)
        print('I am here')
        for element in elements:
            # Checking if the element has multiple components e.g. switcher
            # It is to support element not directly present in FURY
            if isinstance(element.position, list):
                for i, position in enumerate(element.position):
                    self._tab_ui.add_element(tab_id, element.obj[i], position)
            else:
                self._tab_ui.add_element(tab_id, element.obj, element.position)

    def reposition(self, win_size):
        win_width, win_height = win_size
        x_pad = np.rint((win_width - self.__tab_size[0]) / 2)
        self._tab_ui.position = (x_pad, 5)

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
                lambda x: x.tab_type == 'slices_tab'
                and not x.tab_id == active_tab_id, self._tabs
            )
        )

        for slices_tab in slices_tabs:
            slices_tab.update_slices(x_value, y_value, z_value)

    @property
    def tab_ui(self):
        return self._tab_ui


def build_label(text, font_size=16, bold=False, is_horizon_label=True):
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

    if is_horizon_label:
        return HorizonUIElement(True, text, label)
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

    HorizonCombineElement(
        label: HorizonUIElement,
        element(slider): HorizonUIElement)
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
    return HorizonCombineElement(
        slider_label,
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
        The distance between two adjacent options element
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


# TODO: There should be FURY ui element created
def build_switcher(
        items=None,
        label='',
        initial_selection=0,
        on_prev_clicked=lambda _selected_value: None,
        on_next_clicked=lambda _selected_value: None,
        on_value_changed=lambda _selected_idx, _selected_value: None,
):
    """
    Creates horizon theme switcher

    Parameters
    ----------

    items : list
        dictionaries with keys 'label' and 'value'. Label will be used to show
        it to user and value will be used for selection.
    label : str
        label for the switcher.
    initial_selection : int
        index of the selected item initially.
    on_prev_clicked : callback
        method providing a callback when prev value is selected in switcher.
    on_next_clicked : callback
        method providing a callback when next value is selected in switcher.
    on_value_changed : callback
        method providing a callback when either prev or next value selected in
        switcher.

    Returns
    -------

    HorizonCombineElement(
        label: HorizonUIElement,
        element(switcher): HorizonUIElement)

    switcher: consists 'obj' which is an array providing FURY UI elements used.
    """
    # return if there are no items passed
    if items is None:
        warnings.warn('No items passed in switcher')
        return

    num_items = len(items)

    switch_label = build_label(text=label)
    selection_label = build_label(
        text=items[initial_selection]['label'], is_horizon_label=False)

    left_button = ui.Button2D(
            icon_fnames=[('left', read_viz_icons(fname='circle-left.png'))],
            size=(25, 25))
    right_button = ui.Button2D(
            icon_fnames=[('right', read_viz_icons(fname='circle-right.png'))],
            size=(25, 25))

    switcher = HorizonUIElement(True,
                                [initial_selection, items[initial_selection]],
                                [left_button, selection_label, right_button])

    def left_clicked(_i_ren, _obj, _button):
        selected_id = switcher.selected_value[0] - 1
        if selected_id < 0:
            selected_id = num_items - 1
        value_changed(selected_id)
        on_prev_clicked(items[selected_id]['value'])
        on_value_changed(selected_id, items[selected_id]['value'])

    def right_clicked(_i_ren, _obj, _button):
        selected_id = switcher.selected_value[0] + 1
        if selected_id >= num_items:
            selected_id = 0
        value_changed(selected_id)
        on_next_clicked(items[selected_id]['value'])
        on_value_changed(selected_id, items[selected_id]['value'])

    def value_changed(selected_id):
        switcher.selected_value[0] = selected_id
        switcher.selected_value[1] = items[selected_id]['value']
        selection_label.message = items[selected_id]['label']


    left_button.on_left_mouse_button_clicked = left_clicked
    right_button.on_left_mouse_button_clicked = right_clicked

    return HorizonCombineElement(
        switch_label,
        switcher
    )


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
