from typing import Any
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.util import show_ellipsis

fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")

if has_fury:
    from fury import ui
    from fury.data import read_viz_icons


@dataclass
class HorizonUIElement:
    """Dataclass to define properties of horizon ui elements.
    """
    visibility: bool
    selected_value: Any
    obj: Any
    position = (0, 0)
    size = ('auto', 'auto')


class HorizonTab(ABC):
    """Base for different tabs available in horizon.
    """
    def __init__(self):
        self._elements = []

    @abstractmethod
    def build(self, tab_id, tab_ui):
        """Build all the elements under the tab.

        Parameters
        ----------
        tab_id : int
            Id of the tab.
        tab_ui : TabUI
            FURY TabUI object for tabs panel.

        Notes
        -----
        tab_ui will removed once every all tabs adapt new build architecture.
        """

    def register_elements(self, *args):
        """Register elements for rendering.

        Parameters
        ----------
        *args : HorizonUIElement(s)
            Elements to be register for rendering.
        """
        for element in args:
            self._elements.append(element)

    def on_tab_selected(self):
        """Implement if require to update something while the tab becomes
        active.
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Name of the tab.
        """

    @property
    def elements(self):
        """list of underlying FURY ui elements in the tab.
        """
        return self._elements


class TabManager:
    """
    A Manager for tabs of the table panel.

    Attributes
    ----------

    tab_ui : TabUI
        Underlying FURY TabUI object.
    """
    def __init__(self, tabs, win_size, on_tab_changed=lambda actors: None,
                 sync_slices=False, sync_volumes=False, sync_peaks=False):
        num_tabs = len(tabs)
        self._tabs = tabs
        self._synchronize_slices = sync_slices
        self._synchronize_volumes = sync_volumes
        self._synchronize_peaks = sync_peaks

        win_width, _win_height = win_size

        self._tab_size = (1280, 240)
        x_pad = np.rint((win_width - self._tab_size[0]) / 2)

        self._active_tab_id = num_tabs - 1

        self._tab_ui = ui.TabUI(
            position=(x_pad, 5), size=self._tab_size, nb_tabs=num_tabs,
            active_color=(1, 1, 1), inactive_color=(0.5, 0.5, 0.5),
            draggable=True, startup_tab_id=self._active_tab_id)

        self._tab_ui.on_change = self._tab_selected

        self.tab_changed = on_tab_changed

        slices_tabs = list(
            filter(
                lambda x: x.__class__.__name__ == 'SlicesTab', self._tabs
            )
        )
        if not self._synchronize_slices and slices_tabs:
            msg = 'Images are of different dimensions, ' \
                + 'synchronization of slices will not work'
            warnings.warn(msg)

        for tab_id, tab in enumerate(tabs):
            self._tab_ui.tabs[tab_id].title_font_size = 18
            tab.build(tab_id, self._tab_ui)
            if tab.__class__.__name__ == 'SlicesTab':
                tab.on_volume_change = self.synchronize_volumes
            if tab.__class__.__name__ in ['SlicesTab', 'PeaksTab']:
                tab.on_slice_change = self.synchronize_slices
            if tab.__class__.__name__ in ['SlicesTab', 'SurfaceTab']:
                self._render_tab_elements(tab_id, tab.elements)

    def handle_text_overflows(self):
        for tab_id, tab in enumerate(self._tabs):
            self._handle_title_overflow(
                tab.name,
                self._tab_ui.tabs[tab_id].text_block
            )
            if tab.__class__.__name__ == 'SlicesTab':
                self._handle_label_text_overflow(tab.elements)

    def _handle_label_text_overflow(self, elements):
        for element in elements:
            if (not element.size[0] == 'auto' and
                    element.obj.__class__.__name__ == 'TextBlock2D' and
                    isinstance(element.position, tuple)):
                element.obj.message = show_ellipsis(
                    element.selected_value,
                    element.obj.size[0],
                    element.size[0])

    def _handle_title_overflow(self, title_text, title_block):
        """Handle overflow of the tab title and show ellipsis if required.

        Parameters
        ----------
        title_text : str
            Text to be shown on the tab.
        title_block : TextBlock2D
            Fury UI element for holding the title of the tab.
        """
        tab_text = title_text.split('.', 1)[0]
        title_block.message = tab_text
        available_space, _ = self._tab_size
        text_size = title_block.size[0]
        max_width = (available_space / len(self._tabs)) - 15
        title_block.message = show_ellipsis(tab_text, text_size, max_width)

    def _render_tab_elements(self, tab_id, elements):
        for element in elements:
            if isinstance(element.position, list):
                for i, position in enumerate(element.position):
                    self._tab_ui.add_element(tab_id, element.obj[i], position)
            else:
                self._tab_ui.add_element(tab_id, element.obj, element.position)

    def _tab_selected(self, tab_ui):
        if self._active_tab_id == tab_ui.active_tab_idx:
            self._active_tab_id = -1
            return

        self._active_tab_id = tab_ui.active_tab_idx

        current_tab = self._tabs[self._active_tab_id]
        current_tab.on_tab_selected()
        if current_tab.__class__.__name__ in ['SlicesTab', 'SurfaceTab',
                                              'PeaksTab']:
            self.tab_changed(current_tab.actors)

    def reposition(self, win_size):
        """
        Reposition the tabs panel.

        Parameters
        ----------
        win_size : (float, float)
            size of the horizon window.
        """
        win_width, _win_height = win_size
        x_pad = np.rint((win_width - self._tab_size[0]) / 2)
        self._tab_ui.position = (x_pad, 5)

    def synchronize_slices(self, active_tab_id, x_value, y_value, z_value):
        """
        Synchronize slicers for all the images and peaks.

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

        if not self._synchronize_slices and not self._synchronize_peaks:
            return

        for tab in self._get_non_active_tabs(active_tab_id,
                                             ['SlicesTab', 'PeaksTab']):
            tab.update_slices(x_value, y_value, z_value)

    def synchronize_volumes(self, active_tab_id, value):
        """Synchronize volumes for all the images with volumes.

        Parameters
        ----------
        active_tab_id : int
            tab_id of the action performing tab
        value : float
            volume value of the active volume slider

        """

        if not self._synchronize_volumes:
            return

        for slices_tab in self._get_non_active_tabs(active_tab_id):
            slices_tab.update_volume(value)

    def _get_non_active_tabs(self, active_tab_id, types=['SlicesTab']):
        """Get tabs which are not active and slice tabs.

        Parameters
        ----------
        active_tab_id : int
        types : list(str), optional

        Returns
        -------
        list
        """
        return list(
            filter(
                lambda x: x.__class__.__name__ in types
                and not x.tab_id == active_tab_id, self._tabs
            )
        )

    @property
    def tab_ui(self):
        """FURY TabUI object.
        """
        return self._tab_ui


def build_label(text, font_size=16, bold=False, is_horizon_label=False):
    """Simple utility function to build labels.

    Parameters
    ----------
    text : str
    font_size : int, optional
    bold : bool, optional

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
    """Create a horizon theme based disk-knob slider.

    Parameters
    ----------
    initial_value : float, (float, float)
        Initial value(s) of the slider.
    max_value : float
        Maximum value of the slider.
    min_value : float, optional
        Minimum value of the slider.
    length : int, optional
        Length of the slider.
    line_width : int, optional
        Width of the line on which the disk will slide.
    radius : int, optional
        Radius of the disk handle.
    font_size : int, optional
        Size of the text to display alongside the slider (pt).
    text_template : str, callable, optional
        If str, text template can contain one or multiple of the
        replacement fields: `{value:}`, `{ratio:}`.
        If callable, this instance of `:class:LineSlider2D` will be
        passed as argument to the text template function.
    on_moving_slider : callable, optional
        When the slider is interacted by the user.
    on_value_changed : callable, optional
        When value of the slider changed programmatically.
    on_change : callable, optional
        When value of the slider changed.
    label : str, optional
        Label to ui element for slider
    label_font_size : int, optional
        Size of label text to display with slider
    label_style_bold : bool, optional
        Is label should have bold style.

    Return
    ------
    (label: HorizonUIElement, element(slider): HorizonUIElement)
    """

    if is_double_slider and 'ratio' in text_template:
        warnings.warn('Double slider only support values and not ratio')
        return

    slider_label = build_label(
        label,
        font_size=label_font_size,
        bold=label_style_bold,
        is_horizon_label=True
    )

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

    slider.on_moving_slider = on_moving_slider
    slider.on_value_changed = on_value_changed
    slider.on_change = on_change

    slider.default_color = (1., .5, .0)
    slider.track.color = (.8, .3, .0)
    slider.active_color = (.9, .4, .0)
    if not is_double_slider:
        slider.handle.color = (1., .5, .0)
    else:
        slider.handles[0].color = (1., .5, .0)
        slider.handles[1].color = (1., .5, .0)

    return (
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
    """Create horizon theme checkboxes.

    Parameters
    ----------
    labels : list(str), optional
        List of labels of each option.
    checked_labels: list(str), optional
        List of labels that are checked on setting up.
    padding : float, optional
        The distance between two adjacent options element
    font_size : int, optional
        Size of the text font.
    on_change : callback, optional
        When checkbox value changed

    Returns
    -------
    checkbox : HorizonUIElement
    """

    if labels is None or not labels:
        warnings.warn('At least one label needs to be to create checkboxes')
        return

    if checked_labels is None:
        checked_labels = ()

    checkboxes = ui.Checkbox(
        labels=labels,
        checked_labels=checked_labels,
        padding=padding,
        font_size=font_size
    )

    checkboxes.on_change = on_change

    return HorizonUIElement(True, checked_labels, checkboxes)


def build_switcher(
        items=None,
        label='',
        initial_selection=0,
        on_prev_clicked=lambda _selected_value: None,
        on_next_clicked=lambda _selected_value: None,
        on_value_changed=lambda _selected_idx, _selected_value: None,
):
    """Create horizon theme switcher.

    Parameters
    ----------
    items : list, optional
        dictionaries with keys 'label' and 'value'. Label will be used to show
        it to user and value will be used for selection.
    label : str, optional
        label for the switcher.
    initial_selection : int, optional
        index of the selected item initially.
    on_prev_clicked : callback, optional
        method providing a callback when prev value is selected in switcher.
    on_next_clicked : callback, optional
        method providing a callback when next value is selected in switcher.
    on_value_changed : callback, optional
        method providing a callback when either prev or next value selected in
        switcher.

    Returns
    -------
    HorizonCombineElement(
        label: HorizonUIElement,
        element(switcher): HorizonUIElement)

    Notes
    -----
    switcher: consists 'obj' which is an array providing FURY UI elements used.
    """
    if items is None:
        warnings.warn('No items passed in switcher')
        return

    num_items = len(items)

    if initial_selection >= num_items:
        initial_selection = 0

    switch_label = build_label(text=label, is_horizon_label=True)
    selection_label = build_label(
        text=items[initial_selection]['label'])

    left_button = ui.Button2D(
            icon_fnames=[('left', read_viz_icons(fname='circle-left.png'))],
            size=(25, 25))
    right_button = ui.Button2D(
            icon_fnames=[('right', read_viz_icons(fname='circle-right.png'))],
            size=(25, 25))

    switcher = HorizonUIElement(True, [initial_selection,
                                items[initial_selection]['value']],
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

    return (
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
