from dipy.viz.horizon.tab import (HorizonTab, build_slider, build_checkbox)


class ROIsTab(HorizonTab):
    def __init__(self, contour_actors):
        """Initialize interaction tab for ROIs visualization.

        Parameters
        ----------
        contour_actors : list
            list of vtkActor.
        """

        super().__init__()

        self._actors = contour_actors
        self._name = 'ROIs'

        self._tab_id = 0

        self._opacity_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._toggle_opacity)

        self._opacity_slider_label, self._opacity_slider = build_slider(
            initial_value=1, max_value=1., text_template='{ratio:.0%}',
            on_change=self._change_opacity, label='Opacity'
        )

        self.register_elements(
            self._opacity_toggle,
            self._opacity_slider_label, self._opacity_slider
        )

    def _toggle_opacity(self, checkbox):
        """Toggle opacity of all ROIs. Changing the slider.obj.value will
        trigger the _change_opacity function and adjust the opacity.

        Parameters
        ----------
        checkbox : CheckBox2D
        """
        if '' in checkbox.checked_labels:
            self._opacity_slider.selected_value = 1
            self._opacity_slider.obj.value = 1
        else:
            self._opacity_slider.selected_value = 0
            self._opacity_slider.obj.value = 0

    def _change_opacity(self, slider):
        """Change opacity of all ROIs.

        Parameters
        ----------
        slider : LineSlider2D
        """
        opacity = slider.value
        self._opacity_slider.selected_value = slider.value
        for contour in self._actors:
            contour.GetProperty().SetOpacity(opacity)

    def build(self, tab_id, _tab_ui):
        """Position the elements in the tab.

        Parameters
        ----------
        tab_id : int
            Identifier for the tab. Index of the tab in TabUI.
        _tab_ui : TabUI
            Object for Tab from FURY.
        """

        self._tab_id = tab_id

        y_pos = .85
        self._opacity_toggle.position = (.02, y_pos)
        self._opacity_slider_label.position = (.05, y_pos)
        self._opacity_slider.position = (.12, y_pos)

    @property
    def name(self):
        """Title of the tab.

        Returns
        -------
        str
        """
        return self._name

    @property
    def actors(self):
        """Actors controlled by tab.

        Returns
        -------
        list
        """
        return self._actors

    @property
    def tab_id(self):
        """Id of the tab. Reference for Tab Manager to identify the tab.
        Returns
        -------
        int
        """
        return self._tab_id
